# Microduck: steering vectors distilled from specialist policies

Code of the master's thesis described in [docs/Piano_lavoro_tesi_Preda.md](docs/Piano_lavoro_tesi_Preda.md).
How the repository is organized, and why: [REPO_STRUCTURE.md](REPO_STRUCTURE.md).

## Setup

The simulator is the one of the policy cards (CPU MuJoCo + bam servo model). Its packages (mujoco,
mjlab 1.3.0, bam) come from the environment of microduck_rl, at the same commit as phases 0 and 1:

```bash
git clone https://github.com/pollen-robotics/microduck_rl external/microduck_rl
git -C external/microduck_rl checkout 6cd45fc          # hostile-terrain branch, as phases 0 and 1
# if uv is missing
curl -LsSf https://astral.sh/uv/install.sh | sh  # linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # windows

uv sync --project external/microduck_rl
```

Every command then runs in that environment, with this package installed on top:

```bash
uv run --project external/microduck_rl --with-editable . python scripts/<script>.py ...
```

The checkout can live elsewhere: set `MICRODUCK_RL_DIR`. A `microduck_rl/` folder in the repository
root is also found.

### Environment variables

| Variable | Effect |
|---|---|
| `STEERDUCK_STORAGE` | Folder for `models/`, `data/` and `runs/` (e.g. Google Drive on Colab). Default: the repository. |
| `MICRODUCK_RL_DIR` | microduck_rl checkout. Default: `external/microduck_rl`, then `microduck_rl`. |
| `MUJOCO_GL=egl` | Offscreen rendering on a headless machine (videos and preview). |

### Colab

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone https://github.com/mpreda01/microduck_tesi /content/tesi
%cd /content/tesi
!git clone https://github.com/pollen-robotics/microduck_rl external/microduck_rl
!git -C external/microduck_rl checkout --quiet 6cd45fc
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = "/root/.local/bin:" + os.environ["PATH"]
os.environ["UV_HTTP_TIMEOUT"] = "600"
os.environ["STEERDUCK_STORAGE"] = "/content/drive/MyDrive/microduck_thesis"
# offscreen rendering with the NVIDIA driver (GPU runtime), for videos and preview
icd = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(icd):
    os.makedirs(os.path.dirname(icd), exist_ok=True)
    open(icd, "w").write('{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n')
os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
!uv sync --project external/microduck_rl
```

## Phase 1: benchmark of situations

| Step | Command (prefix `uv run --project external/microduck_rl --with-editable .`) |
|---|---|
| Look at the situations | `python scripts/evaluate.py --preview` |
| Quick test | `python scripts/evaluate.py alpha_walking --only flat,friction_0.3 --episodes 2 --no-record` |
| Full benchmark | `python scripts/evaluate.py alpha_walking` |
| With rough-walk-g | `python scripts/evaluate.py alpha_walking rough_walk_g` |
| Table and criterion | `python scripts/report.py --run <run folder> [<run folder> ...]` |

The situations are in [configs/situations.yaml](configs/situations.yaml), one block per family with
the lists of values under `sweep` (edit a list to change slopes, frictions, payloads...). The
simulator and the protocol are in [configs/eval.yaml](configs/eval.yaml).

**Outputs.** `evaluate.py` writes `runs/<YYYYMMDD-HHMM>_evaluate_<name>/` (`config.yaml`, `run.json`,
`log.txt`, `episodes.parquet`, `videos/`) and the per-step data in `data/<run_id>/` (`fields.json` +
`<controller>/<situation>__seed<seed>.npz`). `report.py` writes `results/tables/<run folder>.csv` (several runs joined with `+`): each run keeps its own
table. List the runs used in the thesis in [results/runs_used.yaml](results/runs_used.yaml).

**Timing** (Colab CPUs, 68 situations x 20 episodes): about 32 minutes per policy without the per-step
data; 1-1.5 hours and about 2 GB per policy with them (about 0.5 GB with `--no-activations`).

## Phase 2: specialist policies

Each specialist is `alpha_walking` fine-tuned with PPO on one family of situations, starting from its
training checkpoint (Pollen's `pollen-robotics/microduck-training`, bundle `velocity2/441tzs6d_3750`, set as
`checkpoint` of `alpha_walking` in [configs/policies.yaml](configs/policies.yaml)): actor, critic, normalizers,
action std and optimizer state. Its actor is checked against the ONNX before training. Parameters and families
are in [configs/training.yaml](configs/training.yaml).

| Step | Command (same prefix as phase 1) |
|---|---|
| Quick test of the training | `python scripts/train.py ppo --situation slope_down --num-envs 32 --finetune-iterations 2` |
| Train a specialist (GPU) | `python scripts/train.py ppo --situation slope_down` |
| Train every family (GPU) | `python scripts/train.py ppo --situation all` (one run per family, one after the other) |
| Evaluate it | add `specialist_slope_down: {type: onnx, run: <run folder>}` to [configs/policies.yaml](configs/policies.yaml), then `python scripts/evaluate.py alpha_walking specialist_slope_down --suite fixed` |
| Evaluate an intermediate checkpoint | `python scripts/convert.py checkpoint --run <training run> --iteration 400`, then add the printed `run:` entry to [configs/policies.yaml](configs/policies.yaml) and evaluate it |
| Distance from alpha_walking | `python scripts/action_distance.py --run <evaluation run> --controller alpha_walking [--source <specialist>] [--only <situations>]` |

`train.py` writes `runs/<YYYYMMDD-HHMM>_train_ppo_<family>/` with the checkpoints
`model_<it>.pt`, `<run folder>.onnx` (the latest checkpoint), the tensorboard logs (`tensorboard --logdir runs`)
and `run.json` with the GPU hours.

`action_distance.py` measures how far a specialist is from `alpha_walking` on the states the specialist visited,
without simulating: `alpha_walking` acts on the observations recorded by an evaluation run (so the run needs its
per-step data, i.e. no `--no-record`), and the table `results/tables/action_distance__<run>__<controller>.csv` gives the mean squared
difference between the two actions per situation, without the steps just before a fall, and per joint. About
1 minute for 40 situations x 20 episodes x 2 specialists.

**Weights & Biases.** Training runs are logged to W&B as set in the `wandb` block of
[configs/training.yaml](configs/training.yaml): project, entity, group, tags (the family is added),
mode (`online` / `offline`) and checkpoint upload. The W&B run takes the name of the run folder, and its
config includes the whole `ppo` section of training.yaml. Log in once per machine with `wandb login`
(on Colab or a cluster: `export WANDB_API_KEY=...`). Offline runs are synced later with
`wandb sync runs/<run folder>/wandb/offline-run-*`.

**Hugging Face.** At the end of a training, the specialist is uploaded to the model repo
`<namespace>/<family>-<YYYYMMDD-HHMM>`, e.g. `slope_down-20261022-1943`: the timestamp of the run folder,
which is also the name of the W&B run. The repo holds the ONNX, the last checkpoint, `config.yaml`,
`run.json` and a short model card; `run.json` records the repo id. Settings in the `huggingface` block of
training.yaml (upload on/off, namespace, private). Log in once with `huggingface-cli login` (or
`export HF_TOKEN=...`). To evaluate a policy from its repo, add to configs/policies.yaml
`specialist_slope_down: {type: onnx, hf: [<namespace>/slope_down-<timestamp>, <run folder>.onnx]}`.

## Tests

```bash
uv run --project microduck_rl --with-editable . --with pytest python -m pytest tests
```
