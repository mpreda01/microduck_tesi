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
| Choose the specialist (one command) | `python scripts/choose_checkpoint.py --run <training run> --only "face_*_downhill,pyramid_*"`, then add the printed entry to [configs/policies.yaml](configs/policies.yaml) |
| Upload a specialist to Hugging Face | `python scripts/upload_to_hf.py --run <choose, convert or training run>` |
| Only the candidates, no simulation | `python scripts/choose_checkpoint.py --run <training run> --shortlist --states <evaluation run> --source <specialist> --only "face_*_downhill,pyramid_*"` |
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

`choose_checkpoint.py` chooses the specialist among the checkpoints of a training run, in one command:
1. it evaluates the last checkpoint on the given situations (`--only` takes names or patterns), recording the
   per-step data;
2. it shortlists the candidates. For each `model_<it>.pt` it takes the training curves averaged around that
   iteration (reward, % of episodes ended in a fall, terrain ladder, action std) and its distance from
   `alpha_walking` on those states. A checkpoint is on the plateau when it has reached the whole ladder and its fall
   share is close to the best one (rule: `ppo.checkpoint_choice` in [configs/training.yaml](configs/training.yaml)).
   The candidates are the plateau checkpoint closest to `alpha_walking`, the middle one and the last;
3. it evaluates the other candidates on the same episodes;
4. it measures each candidate's distance from `alpha_walking` on its own states;
5. it applies the rule: fewest falls, and among equivalent candidates (overlapping Wilson intervals of falls and
   successes) the one closest to `alpha_walking`.

With `--shortlist` it stops after step 2. With `--states <evaluation run> --source <controller>` as well, step 2
uses the states of an existing evaluation: nothing is simulated (about 30 s).

It writes `runs/<YYYYMMDD-HHMM>_choose_<training run>/`, holding:
- the ONNX of each candidate, and the chosen one as `<run folder>.onnx`;
- `checkpoints.csv` (the curves) and `choice.csv` (falls, successes, Wilson intervals, distance, `chosen`);
- `config.yaml`, `run.json` and `log.txt`.

It prints the `policies.yaml` entry for the chosen checkpoint. The two evaluations are ordinary evaluation runs. With
`--episodes 2` it is a quick test; with the default 20 episodes it takes as long as evaluating three policies on the
family's situations.

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
`export HF_TOKEN=...`). `scripts/upload_to_hf.py --run <run>` uploads later, with the settings saved in the training
run:
- given a training run, its last checkpoint, as above;
- given a `convert` or `choose` run, only the checkpoint it converted or chose, to
  `<namespace>/<family>-<timestamp>-it<it>`. The repo holds `<family>-<timestamp>-it<it>.onnx`, `model_<it>.pt`, the
  training's `config.yaml` and `run.json`, and a model card naming the source run; the other candidates stay local.

It prints the `hf:` entry for `policies.yaml`. To evaluate a policy from its repo, add to configs/policies.yaml
`specialist_slope_down: {type: onnx, hf: [<namespace>/slope_down-<timestamp>, <run folder>.onnx]}`.

## Phase 3: library of corrections

Step 1 of the plan: the SAE and the linear probes, on the activations that `alpha_walking` recorded in every
situation (an evaluation run without `--no-record` and `--no-activations`). Parameters in the `sae` section of
[configs/training.yaml](configs/training.yaml).

| Step | Command (same prefix as phase 1) |
|---|---|
| SAE and probes | `python scripts/train.py sae --run 20261028-1239_evaluate_alpha_walking_all` |
| Check that the policy still walks with the SAE's reconstruction | add the printed `alpha_walking+sae_h2: {type: sae, base: alpha_walking, run: <run folder>}` to [configs/policies.yaml](configs/policies.yaml), then `python scripts/evaluate.py alpha_walking+sae_h2 --no-activations` and compare with `report.py --run <that run> <alpha_walking's run>` |

`train.py sae` writes `runs/<YYYYMMDD-HHMM>_train_sae_<layer>/` with:
- `sae.pt`;
- `probes.csv`: R² of a linear probe on the test episodes, per layer and quantity;
- `features.csv`: each dictionary element named after the quantity it correlates with most;
- `error_by_situation.csv`: the reconstruction error of each situation on the test episodes;
- `run.json`: test FVU and the share of dictionary elements never active.

It takes about 2 minutes on CPU (275k steps).

Step 2 of the plan: the constant steering vector of each specialist, with DAgger, distilled on the seeds of
[configs/eval.yaml](configs/eval.yaml) (0) and evaluated on another one (42). Parameters in the `distill` section of
[configs/training.yaml](configs/training.yaml): the specialist and the situations of each family.

| Step | Command (same prefix as phase 1) |
|---|---|
| Distil the vector (h1, h2, h3 or all) | `python scripts/train.py distill --situation slope_down --layer h2` |
| Evaluate on unseen episodes | add the printed `alpha_walking+const_h2_slope_down: {type: adapter, ...}` to [configs/policies.yaml](configs/policies.yaml), then `python scripts/evaluate.py alpha_walking slope_down-20261027-2240-it400 alpha_walking+const_h2_slope_down --only "face_*_downhill,pyramid_*,flat" --seeds 42 --no-record` and `report.py --run <that run>` |

`train.py distill` writes `runs/<YYYYMMDD-HHMM>_train_distill_alpha_walking+const_<layer>_<situation>/` with
`adapter.pt` (the DAgger iteration with most successes, then fewest falls), `adapter_it<k>.pt` and `curves.csv` (steps,
fit error, successes and falls of each iteration). Each rollout is an evaluation run of its own.

## Tests

```bash
uv run --project microduck_rl --with-editable . --with pytest python -m pytest tests
```
