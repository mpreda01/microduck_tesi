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
    ft = ppo["finetune"]
    cfg.max_iterations = ft["iterations"]
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
    """The loaded optimizer state carries the learning rate the checkpoint ended with: replace it with the
    fine-tuning one (the fixed schedule would otherwise keep it)."""
    alg.schedule, alg.learning_rate = schedule, learning_rate
    for group in alg.optimizer.param_groups:
        group["lr"] = learning_rate


def upload_to_huggingface(run_dir: Path, family: str, ppo: dict) -> str:
    """Upload a trained specialist to the Hugging Face model repo <namespace>/<family>-<YYYYMMDD-HHMM>.
    The timestamp is the one of the run folder (<YYYYMMDD-HHMM>_train_ppo_<family>), which is also the name
    of the W&B run: repo, local run and W&B run point to the same training. -> the repo id."""
    from huggingface_hub import HfApi

    api = HfApi()
    hf = ppo["huggingface"]
    namespace = hf["namespace"] or api.whoami()["name"]
    timestamp = run_dir.name.split("_")[0]
    repo_id = f"{namespace}/{family}-{timestamp}"

    onnx = run_dir / f"{run_dir.name}.onnx"
    last_checkpoint = max(run_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    files = [onnx, last_checkpoint, run_dir / "config.yaml", run_dir / "run.json"]
    model_card = (
        f"# {family} specialist ({timestamp})\n\n"
        f"Microduck walking policy: `{ppo['init_controller']}` fine-tuned with PPO on the `{family}` family.\n\n"
        f"- training run and W&B run: `{run_dir.name}` "
        f"(W&B project `{ppo['wandb']['project']}`, group `{ppo['wandb']['group']}`)\n"
        f"- `{onnx.name}`: the policy, 61 observations -> 14 actions\n"
        f"- `{last_checkpoint.name}`: the last rsl_rl checkpoint (actor and critic)\n"
        f"- `config.yaml`, `run.json`: configuration and manifest of the run\n")

    api.create_repo(repo_id, private=hf["private"], exist_ok=True)
    for path in files:
        api.upload_file(path_or_fileobj=path, path_in_repo=path.name, repo_id=repo_id)
    api.upload_file(path_or_fileobj=model_card.encode(), path_in_repo="README.md", repo_id=repo_id)
    return repo_id


def train_specialist(family: str, ppo: dict, run_dir: Path, device: str):
    """Fine-tune ppo['init_controller'] on one family. Checkpoints go to run_dir (model_<it>.pt) and the
    latest one also as <run_id>.onnx, the file configs/policies.yaml points to with `run: <run_id>`."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner

    from steerduck.policy import actor_onnx_difference, resolve, resolve_checkpoint

    configure_torch_backends()
    set_wandb_environment(ppo["wandb"], family, run_dir)
    env_cfg, rl_cfg = make_env_cfg(ppo["families"][family], ppo), make_rl_cfg(ppo)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=rl_cfg.clip_actions)
    # our training.yaml section rides along in the runner config, which the W&B writer uploads as train_cfg
    train_cfg = asdict(rl_cfg) | {"steerduck": {"family": family, **ppo}}
    runner = MicroduckOnPolicyRunner(env, train_cfg, str(run_dir), device)

    # start: the training checkpoint of the evaluated policy (actor, critic, normalizers, action std, optimizer
    # state), its actor checked against the ONNX. Not its iteration: this run counts its own from 0
    onnx_path = resolve(ppo["init_controller"])["path"]
    checkpoint = resolve_checkpoint(ppo["init_controller"])
    runner.load(checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": True}, map_location=device)
    diff = actor_onnx_difference(runner.alg.actor, onnx_path)
    if diff >= 1e-5:
        raise ValueError(f"the actor of {checkpoint} differs from {onnx_path} by {diff:.1e}")
    log.info(f"loaded {checkpoint}: largest difference of its actor from the ONNX {diff:.1e}")

    ft = ppo["finetune"]
    _set_learning_rate(runner.alg, ft["schedule"], ft["learning_rate"])
    runner.learn(num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()
    return diff
