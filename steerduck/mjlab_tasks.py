"""The situations of situations.yaml as mjlab training tasks, and the PPO fine-tuning of a specialist
(REPO_STRUCTURE.md, section 4).

The terrains are built by scene.make_subterrain, the same function as the evaluation, so a situation has
the same geometry in training (mjlab, GPU) and in evaluation (CPU MuJoCo + bam).
"""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg

from steerduck.scene import make_subterrain

log = logging.getLogger("steerduck")


def _is_range(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)


@dataclass(kw_only=True)
class LadderTerrainCfg(SubTerrainCfg):
    """A terrain of situations.yaml whose values written as [easy, hard] grow linearly with the difficulty
    of the ladder row (mjlab's curriculum: one column per terrain, one row per difficulty)."""
    situation: dict

    def function(self, difficulty, spec, rng):
        s = {k: v[0] + difficulty * (v[1] - v[0]) if _is_range(v) else v for k, v in self.situation.items()}
        sizes = {"patch_size_m": self.size[0], "face_slope_size_m": self.size[0], "strip_width_m": self.size[1]}
        sub, _ = make_subterrain(s, sizes)
        sub.size = self.size
        return sub.function(difficulty, spec, rng)


def make_env_cfg(terrains: list[dict], ppo: dict):
    """microduck_rl's hostile-terrain walking task (reward, fine-tune mode, ladder promotion) with the
    terrain ladder of one family, plus a share of flat patches so that flat walking is not forgotten."""
    from mjlab_microduck.tasks.microduck_velocity_hostile_env_cfg import make_microduck_velocity_hostile_env_cfg

    cfg = make_microduck_velocity_hostile_env_cfg(finetune=True, **ppo["reward"])
    t = ppo["terrain"]
    share = (1.0 - t["flat_proportion"]) / len(terrains)
    sub_terrains = {f"{s['terrain']}_{i}": LadderTerrainCfg(proportion=share, situation=s) for i, s in enumerate(terrains)}
    sub_terrains["flat"] = BoxFlatTerrainCfg(proportion=t["flat_proportion"])
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(t["patch_size_m"], t["patch_size_m"]), border_width=t["border_width_m"], num_rows=t["ladder_rows"],
        num_cols=len(sub_terrains), curriculum=True, sub_terrains=sub_terrains, add_lights=False)
    cfg.scene.num_envs = ppo["num_envs"]
    cfg.seed = ppo["seed"]
    return cfg


def make_rl_cfg(ppo: dict):
    """microduck_rl's PPO configuration of the hostile task, with the fine-tuning values of training.yaml."""
    from mjlab_microduck.tasks.microduck_velocity_hostile_env_cfg import MicroduckHostileRlCfg

    cfg = copy.deepcopy(MicroduckHostileRlCfg)
    cfg.seed, cfg.save_interval = ppo["seed"], ppo["save_interval"]
    cfg.logger = "wandb" if ppo["wandb"]["enabled"] else "tensorboard"
    cfg.wandb_project, cfg.upload_model = ppo["wandb"]["project"], ppo["wandb"]["upload_model"]
    cfg.max_iterations = ppo["critic_warmup"]["iterations"] + ppo["finetune"]["iterations"]
    ft = ppo["finetune"]
    cfg.algorithm.learning_rate, cfg.algorithm.schedule = ft["learning_rate"], ft["schedule"]
    cfg.algorithm.desired_kl, cfg.algorithm.entropy_coef = ft["desired_kl"], ft["entropy_coef"]
    return cfg


def set_wandb_environment(wandb: dict, family: str, run_dir: Path):
    """rsl_rl's W&B writer takes only the project from its config and leaves entity, tags, group, mode and
    local folder to wandb.init, which reads them from the WANDB_* environment variables: set them from
    training.yaml. The W&B run is named after the run folder, and its local files stay inside it."""
    env = {"WANDB_ENTITY": wandb["entity"], "WANDB_RUN_GROUP": wandb["group"], "WANDB_MODE": wandb["mode"],
           "WANDB_TAGS": ",".join([*wandb["tags"], family]), "WANDB_DIR": run_dir}
    os.environ.update({k: str(v) for k, v in env.items() if v})


def _set_learning_rate(alg, schedule: str, learning_rate: float):
    alg.schedule, alg.learning_rate = schedule, learning_rate
    for group in alg.optimizer.param_groups:
        group["lr"] = learning_rate


def stage_critic_warmup(alg, warmup: dict, finetune: dict):
    """The critic learns alone for the first warmup['iterations'] PPO updates: the actor (weights, normalizer
    and action std) gets no gradient, and the learning rate is fixed (the adaptive schedule reads the KL
    divergence of the actor, which does not move). Then the actor is released and the run goes on with the
    fine-tuning schedule. Done by wrapping alg.update, so that both stages are one rsl_rl run and one log."""
    if warmup["iterations"] == 0:
        return
    alg.actor.requires_grad_(False)
    _set_learning_rate(alg, "fixed", warmup["learning_rate"])
    update, done = alg.update, 0

    def staged_update():
        nonlocal done
        losses = update()
        done += 1
        if done == warmup["iterations"]:
            alg.actor.requires_grad_(True)
            _set_learning_rate(alg, finetune["schedule"], finetune["learning_rate"])
            log.info(f"critic warm-up done after {done} iterations: fine-tuning actor and critic")
        return losses

    alg.update = staged_update


def train_specialist(family: str, ppo: dict, run_dir: Path, device: str):
    """Fine-tune ppo['init_controller'] on one family. Checkpoints go to run_dir (model_<it>.pt) and the
    latest one also as <run_id>.onnx, the file configs/policies.yaml points to with `run: <run_id>`."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner

    from steerduck.policy import actor_onnx_difference, onnx_to_rsl_rl_actor, resolve

    configure_torch_backends()
    set_wandb_environment(ppo["wandb"], family, run_dir)
    env_cfg, rl_cfg = make_env_cfg(ppo["families"][family], ppo), make_rl_cfg(ppo)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=rl_cfg.clip_actions)
    # our training.yaml section rides along in the runner config, which the W&B writer uploads as train_cfg
    train_cfg = asdict(rl_cfg) | {"steerduck": {"family": family, **ppo}}
    runner = MicroduckOnPolicyRunner(env, train_cfg, str(run_dir), device)

    # start: the actor of the evaluated policy, checked against its ONNX; critic and optimizer are new
    onnx_path = resolve(ppo["init_controller"])["path"]
    init = run_dir / "init_actor.pt"
    torch.save({"actor_state_dict": onnx_to_rsl_rl_actor(onnx_path, ppo["action_std"]), "iter": 0, "infos": None}, init)
    runner.load(str(init), load_cfg={"actor": True}, map_location=device)
    diff = actor_onnx_difference(runner.alg.actor, onnx_path)
    if diff >= 1e-5:
        raise ValueError(f"the rebuilt actor differs from {onnx_path} by {diff:.1e}")
    log.info(f"actor rebuilt from {onnx_path}: largest difference from the ONNX {diff:.1e}")

    stage_critic_warmup(runner.alg, ppo["critic_warmup"], ppo["finetune"])
    runner.learn(num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()
    return diff
