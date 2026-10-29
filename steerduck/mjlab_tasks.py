"""The situations of situations.yaml as mjlab training tasks, and the PPO fine-tuning of a specialist
(REPO_STRUCTURE.md, section 4).

The terrains are built by scene.make_subterrain, the same function as the evaluation, so a situation has
the same geometry in training (mjlab, GPU) and in evaluation (CPU MuJoCo + bam).
"""
from __future__ import annotations

import copy
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg

from steerduck.scene import make_subterrain

log = logging.getLogger("steerduck")

HEADINGS_RAD = {"downhill": 0.0, "uphill": math.pi}   # world heading of the walk along the +x face of a pyramid


def _is_range(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)


def face_start(pyramid, direction: str, margin_m: float) -> np.ndarray:
    """Start on the centre line of the +x face of a face_slope pyramid (scene.FlatFacePyramidTerrainCfg), in
    the patch frame: margin_m past the top platform (downhill) or margin_m from the edge (uphill). The ground
    there is tan(slope) x (distance from the edge); one hfield cell more, so that the robot never starts
    inside the ground and drops at most one cell."""
    half_m = pyramid.size[0] / 2
    centre_m = pyramid.platform_width / 2 + margin_m if direction == "downhill" else half_m - margin_m
    z_m = math.tan(math.radians(pyramid.slope_deg)) * (half_m - centre_m + pyramid.horizontal_scale)
    return np.array([half_m + centre_m, pyramid.size[1] / 2, z_m])


@dataclass(kw_only=True)
class LadderTerrainCfg(SubTerrainCfg):
    """A terrain of situations.yaml whose values written as [easy, hard] grow linearly with the difficulty
    of the ladder row (mjlab's curriculum: one column per terrain, one row per difficulty). A face_slope
    moves the spawn origin from the top to the start of its direction (face_start)."""
    situation: dict
    start_margin_m: float

    def function(self, difficulty, spec, rng):
        s = {k: v[0] + difficulty * (v[1] - v[0]) if _is_range(v) else v for k, v in self.situation.items()}
        sizes = {"patch_size_m": self.size[0], "face_slope_size_m": self.size[0], "strip_width_m": self.size[1]}
        sub, _ = make_subterrain(s, sizes)
        sub.size = self.size
        output = sub.function(difficulty, spec, rng)
        if s["terrain"] == "face_slope":
            output.origin = face_start(sub, s["direction"], self.start_margin_m)
        return output


def family_heading(terrains: list[dict]) -> float:
    """World heading [rad] the robot keeps on a family: the one of its direction, shared by all its terrains."""
    directions = {s.get("direction") for s in terrains}
    if len(directions) != 1 or not directions <= HEADINGS_RAD.keys():
        raise ValueError(f"the terrains of a family need one direction among {list(HEADINGS_RAD)}, "
                         f"got {sorted(map(str, directions))}")
    return HEADINGS_RAD[directions.pop()]


def set_straight_walking(cfg, heading_rad: float, command: dict):
    """The command of the evaluation instead of microduck_rl's random one: forward speed only, heading held
    at heading_rad, no standing or turning in place (the curriculum that sets the standing share is removed).
    The robot starts on the line of the walk (no offset along x) with the same heading +- the jitter, so it
    starts at the height of the spawn origin and walks only downhill or only uphill."""
    twist = cfg.commands["twist"]
    max_yaw = command["max_yaw_rate_rad_s"]
    twist.ranges.lin_vel_x, twist.ranges.lin_vel_y = tuple(command["lin_vel_x_m_s"]), (0.0, 0.0)
    twist.ranges.ang_vel_z, twist.ranges.heading = (-max_yaw, max_yaw), (heading_rad, heading_rad)
    twist.heading_command, twist.rel_heading_envs = True, 1.0
    twist.heading_control_stiffness = command["heading_stiffness"]
    twist.rel_standing_envs = twist.rel_turn_in_place_envs = twist.rel_forward_envs = twist.rel_world_envs = 0.0
    cfg.curriculum.pop("standing_envs")
    jitter = math.radians(command["heading_jitter_deg"])
    pose = cfg.events["reset_base"].params["pose_range"]
    pose["x"], pose["yaw"] = (0.0, 0.0), (heading_rad - jitter, heading_rad + jitter)


def make_env_cfg(terrains: list[dict], ppo: dict):
    """microduck_rl's hostile-terrain walking task (reward, fine-tune mode, ladder promotion) with the
    terrain ladder of one family, plus a share of flat patches so that flat walking is not forgotten, and
    the straight walk of the evaluation in the family's direction."""
    from mjlab_microduck.tasks.microduck_velocity_hostile_env_cfg import make_microduck_velocity_hostile_env_cfg

    cfg = make_microduck_velocity_hostile_env_cfg(finetune=True, **ppo["reward"])
    set_straight_walking(cfg, family_heading(terrains), ppo["command"])
    t = ppo["terrain"]
    share = (1.0 - t["flat_proportion"]) / len(terrains)
    sub_terrains = {f"{s['terrain']}_{i}": LadderTerrainCfg(proportion=share, situation=s,
                                                            start_margin_m=t["start_margin_m"])
                    for i, s in enumerate(terrains)}
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


def upload_to_huggingface(run_dir: Path, family: str, ppo: dict, checkpoint_run: Path | None = None,
                          iteration: int | None = None) -> str:
    """Upload a trained specialist to the Hugging Face model repo <namespace>/<family>-<YYYYMMDD-HHMM>.
    The timestamp is the one of the run folder (<YYYYMMDD-HHMM>_train_ppo_<family>), which is also the name
    of the W&B run: repo, local run and W&B run point to the same training. With checkpoint_run (a convert or
    choose run of this training, whose <run_id>.onnx is model_<iteration>.pt): that checkpoint instead of the
    last one, to <namespace>/<family>-<YYYYMMDD-HHMM>-it<iteration>, with the files of checkpoint_run in a folder
    of the same name. -> the repo id."""
    from huggingface_hub import HfApi

    api = HfApi()
    hf = ppo["huggingface"]
    namespace = hf["namespace"] or api.whoami()["name"]
    timestamp = run_dir.name.split("_")[0]
    if checkpoint_run is None:
        onnx = run_dir / f"{run_dir.name}.onnx"
        checkpoint = max(run_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
        repo_id, extra, which = f"{namespace}/{family}-{timestamp}", [], "the last"
    else:
        onnx, checkpoint = checkpoint_run / f"{checkpoint_run.name}.onnx", run_dir / f"model_{iteration}.pt"
        repo_id, which = f"{namespace}/{family}-{timestamp}-it{iteration}", f"the iteration {iteration}"
        extra = [p for p in checkpoint_run.iterdir() if p.is_file() and p != onnx]

    files = [onnx, checkpoint, run_dir / "config.yaml", run_dir / "run.json"]
    model_card = (
        f"# {family} specialist ({timestamp}{f', iteration {iteration}' if checkpoint_run else ''})\n\n"
        f"Microduck walking policy: `{ppo['init_controller']}` fine-tuned with PPO on the `{family}` family.\n\n"
        f"- training run and W&B run: `{run_dir.name}` "
        f"(W&B project `{ppo['wandb']['project']}`, group `{ppo['wandb']['group']}`)\n"
        f"- `{onnx.name}`: the policy, 61 observations -> 14 actions\n"
        f"- `{checkpoint.name}`: {which} rsl_rl checkpoint (actor and critic)\n"
        f"- `config.yaml`, `run.json`: configuration and manifest of the training run\n"
        + (f"- `{checkpoint_run.name}/`: the run that made the checkpoint into ONNX\n" if checkpoint_run else ""))

    api.create_repo(repo_id, private=hf["private"], exist_ok=True)
    for path in files:
        api.upload_file(path_or_fileobj=path, path_in_repo=path.name, repo_id=repo_id)
    for path in extra:
        api.upload_file(path_or_fileobj=path, path_in_repo=f"{checkpoint_run.name}/{path.name}", repo_id=repo_id)
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
