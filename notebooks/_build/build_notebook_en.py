import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]

with open(os.path.join(HERE, "eval_scenario.py"), encoding="utf-8") as f:
    script = f.read()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md(r"""
# Microduck: evaluating a walking policy over a suite of conditions

Evaluates the policy you choose on a **suite of fixed conditions** from Phase 1 of the work plan, with the same number of episodes for each condition (100 by default):

- **flat ground** (reference), with random commands and with a constant forward walk;
- **constant slopes walked downhill and uphill** at several angles, also beyond the training range, on the pyramid terrain of the rough-terrain training, with the heading held;
- **terrain types** from section 4.6 (low stairs, uneven ground, waves, ...);
- **reduced friction**, set on **both the floor and the feet**: MuJoCo uses the larger of the two coefficients, so changing only one of them would have no effect;
- **payloads** attached to the trunk, which **shift the centre of mass** (mass and position are chosen per condition).

Each condition runs with the same seed, so two policies evaluated with the same settings see the same conditions (paired comparison).

**What gets saved**, for each condition, in `MyDrive/microduck_thesis/eval/<suite_name>/<condition>/`:

| File | Content | Why this format |
|---|---|---|
| `summary.json` | configuration, software versions, aggregate metrics (falls per 100 episodes with 95% confidence interval, velocity error, power) and one row per episode with outcome and scenario | small and readable, opens anywhere |
| `steps.npz` | one row per control step: observations, hidden-layer activations, action, body motion, feet, torques, power, "falls within 2 s" label | 100 episodes of 20 s are about 100,000 steps of ~1,200 numbers: as JSON that would be several GB and very slow to read. Compressed NumPy with float16 activations is the format proposed in section 4.9 |
| `episode_video.mp4` | video of one of the evaluated episodes (the first one), until it ends or falls | to check by eye what the numbers describe |

Plus `suite_summary.json` in the suite folder: one row per condition, used for the results table and the policy comparison.

**Before you start:** *Runtime → Change runtime type → T4 GPU*. Run cells 1-7 once per session, then cell 8 to run the suite.
"""),
    md("## 1. GPU check"),
    code("!nvidia-smi"),
    md("## 2. Installation"),
    code(r"""
import os

!curl -LsSf https://astral.sh/uv/install.sh | sh
os.environ["PATH"] = "/root/.local/bin:" + os.environ["PATH"]
os.environ["UV_HTTP_TIMEOUT"] = "600"   # avoids timeouts on the large CUDA packages

if not os.path.exists("/content/microduck_rl"):
    !git clone https://github.com/pollen-robotics/microduck_rl /content/microduck_rl
%cd /content/microduck_rl
!uv sync
!uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"
"""),
    md(r"""
## 3. Offscreen rendering (for the videos)

Colab has no display: MuJoCo renders the video frames on the GPU through EGL. This cell registers the NVIDIA EGL driver if it is missing (the same fix used by the official MuJoCo notebooks) and selects EGL for the evaluation script.
"""),
    code(r"""
NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    os.makedirs(os.path.dirname(NVIDIA_ICD_CONFIG_PATH), exist_ok=True)
    with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
        f.write('{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n')

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
"""),
    md("## 4. Google Drive\n\nGrant access when asked. Results go to `DRIVE_DIR` (cell 7)."),
    code(r"""
from google.colab import drive
drive.mount("/content/drive")
"""),
    md(r"""
## 5. Choose the policy

Set `POLICY` to one of the names in `POLICIES`. A Hugging Face entry is `(repo, file)`; a local or Drive file is just its path. To add a policy, add a line.

If a download fails with 401/404 the repo is private: run `!uv run huggingface-cli login` with a token that has access. To check the file names in a repo: `!uv run python -c "from huggingface_hub import list_repo_files; print(list_repo_files('<repo>'))"`.
"""),
    code(r"""
POLICIES = {
    "alpha_walking": ("pollen-robotics/microduck-policies", "alpha_walking.onnx"),
    "rough_walk_g": ("RemiFabre/microduck-rough-walk-g", "policy.onnx"),
    # "my_policy": "/content/drive/MyDrive/policies/my_policy.onnx",
}
POLICY = "alpha_walking"

source = POLICIES[POLICY]
if isinstance(source, str):
    POLICY_PATH = source
else:
    repo, filename = source
    !uv run python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('{repo}', '{filename}', local_dir='policies/{POLICY}'))"
    POLICY_PATH = os.path.abspath(os.path.join("policies", POLICY, filename))
print("policy:", POLICY, "->", POLICY_PATH, "| found:", os.path.exists(POLICY_PATH))
"""),
    md(r"""
## 6. The evaluation script

It evaluates **one condition**: builds the Microduck walking environment (`make_microduck_velocity_env_cfg`) with that condition, converts the ONNX policy into a PyTorch `nn.Sequential` and **checks** it against onnxruntime on real observations (section 4.5; it stops if the difference exceeds 1e-4), records activations with a forward hook after each ELU (section 4.10), runs the episodes in parallel and saves the data. Cell 8 calls it once per condition.

**Video:** during the first round, at every step, the state of robot copy 0 is copied from the GPU simulator into a CPU MuJoCo model of the same scene and rendered with a camera that follows the trunk. The video therefore shows exactly one of the evaluated episodes (episode 0), with the same terrain, ramp and payload, and stops when that episode ends. If rendering fails, the evaluation goes on without the video and a warning is written in `summary.json`.

How each part of a condition is implemented:

| Part | Implementation |
|---|---|
| Slope with direction | The pyramid heightfield of the rough-terrain training (`HfPyramidSlopedTerrainCfg`) at a fixed angle, as one 40 × 40 m pyramid so that a face is about 19 m long. mjlab builds the pyramid as H·tent(x)·tent(y), so the slope is exactly the nominal one only on the centre lines: the robot is placed on the +x centre line with no random offset. **Uphill and downhill are the same task in opposite directions:** both start in the middle of the face (10 m from the centre, at the computed surface height), with about 9 m available each way; downhill faces the edge of the pyramid, uphill faces the top. Constant forward command with heading hold |
| Heading hold | mjlab's heading controller: every step the yaw-rate command is `clip(heading error, ±1 rad/s)`, with the spawn yaw as target, like an operator correcting with a joystick. Without it the policies drift (`rough_walk_g` up to ~0.3 rad/s, always to the left) and stop walking straight up or down the slope. It changes the command the policy receives, so it must be stated when reporting |
| Ramp (not used) | `FlatRampTerrainCfg` (one long tilted box, from the RollerSlope task) made both policies fall 75-85% of the time already at 2.5°, against 0% on a 5° pyramid face: an artefact of that terrain, so it is not used |
| Terrain type | mjlab terrain generator, 4 × 4 patches of 8 × 8 m, all at the chosen difficulty |
| Friction | the friction of every terrain geom **and** of the two foot geoms is set to the same value; the training foot-friction randomization is switched off for that condition |
| Payload | a rigid body welded to the trunk, with the given mass, placed at the given offset from the trunk centre of mass (x forward, y left, z up): MuJoCo recomputes total mass, centre of mass and inertia |
| Everything else | as in training (centre-of-mass, mass, encoder and IMU randomization); pushes off unless `PUSHES = True` |

The actual foot friction, terrain friction and trunk mass of every episode are read back from the simulator and written in `summary.json`, so you can check that the condition was applied.
"""),
    code("%%writefile eval_scenario.py\n" + script),
    code("!uv run python eval_scenario.py --list"),
    md(r"""
## 7. Define the suite

Edit the lists to choose the conditions. Every condition gets `EPISODES_PER_CONDITION` episodes. Names are built automatically, e.g. `slope_10deg_downhill`, `friction_0.3`, `payload_160g_back`.

`EXTRA_CONDITIONS` takes custom or combined conditions as dictionaries with these keys (all optional except `name`):

| Key | Meaning |
|---|---|
| `name` | folder name of the condition |
| `terrain` | a name from the list above (default `flat`); `plane` = exactly the flat environment `alpha_walking` was trained in (infinite plane, default contacts and solver), while `flat` is a generated flat terrain with the softened contacts used by all the other terrains |
| `difficulty` | 0-1, position inside the terrain parameter range (default 1.0) |
| `slope_deg` | slope angle for `terrain: "slope"` (the pyramid) |
| `direction` | with `terrain: "slope"`: `"downhill"` or `"uphill"` along the centre line of one face (sets spawn position, height and yaw) |
| `tile_size_m` | pyramid size for directional slopes (default 40) |
| `slope_start_m` | distance of the start from the pyramid centre, the same for uphill and downhill (default: a quarter of the pyramid size, i.e. the middle of the face) |
| `heading_deg` | fixed spawn yaw on other terrains; omitted = random |
| `command` | `"random"` (training distribution, default) or `{"vx": 0.3}` for a constant forward walk |
| `hold_heading` | with a constant command: `True` keeps the spawn heading with mjlab's heading controller |
| `friction` | friction of floor and feet; omitted = training randomization |
| `payload` | `{"mass_kg": 0.16, "offset_m": [x, y, z]}`, offset from the trunk centre of mass |
| `terrain_params` | overrides of the terrain parameters |

`SUITE_NAME` is fixed (no timestamp) on purpose: if Colab disconnects, re-running cell 8 skips the conditions already saved. Change it (or `SEED`) to start a new suite.

**Drive space:** with `SAVE_STEPS = True` each condition takes roughly 150-250 MB. Activations are needed for the policy you study with SAE and steering; for comparison-only runs (e.g. `rough_walk_g`) set `SAVE_STEPS = False`.
"""),
    code(r"""
# ======================= SUITE =======================
EPISODES_PER_CONDITION = 100
EPISODE_LENGTH_S = 20.0
FORWARD_SPEED = 0.3                    # m/s, constant forward command for slopes and "flat_forward";
                                       # the policies reach about half of it (0.1 m/s is a dead zone)
HOLD_HEADING = True                    # keep the spawn heading on slopes and "flat_forward"

SLOPES_DEG = [0, 2.5, 5, 7.5, 10, 15]  # pyramid face; 0 = control (flat heightfield, same code)
                                       # training on rough terrain reaches ~5.7 deg
SLOPE_DIRECTIONS = ["downhill", "uphill"]
SLOPE_TILE_SIZE = 40.0                 # m: faces about 19 m long; the start is in the middle of a face,
                                       # with ~9 m available both uphill and downhill

FRICTIONS = [0.6, 0.4, 0.3, 0.2, 0.1]  # floor and feet; training feet friction is 0.7-1.3

PAYLOADS = {                           # name: (mass kg, offset from trunk CoM [x fwd, y left, z up] m)
    "payload_80g": (0.08, (0.0, 0.0, 0.03)),
    "payload_160g": (0.16, (0.0, 0.0, 0.03)),
    "payload_160g_back": (0.16, (-0.04, 0.0, 0.03)),
    "payload_160g_front": (0.16, (0.04, 0.0, 0.03)),
    "payload_160g_side": (0.16, (0.0, 0.04, 0.03)),
}

TERRAINS = {                           # name: difficulty (0-1)
    "slope": 1.0,                      # pyramid ~10 deg, random walking directions (8 m patches)
    "pyramid_stairs": 1.0,             # steps up to 1.5 cm
    "random_stairs": 1.0,
    "open_stairs": 1.0,
    "random_grid": 1.0,                # cobblestones up to 1 cm
    "random_uniform": 1.0,             # bumps up to 2 cm
    "waves": 1.0,
    "perlin": 1.0,
    "tilted_grid": 1.0,
    "stepping_stones": 1.0,            # these last five use mjlab defaults or guessed parameters:
    "narrow_beams": 1.0,               # check "terrain_params_used" and "warnings" in summary.json
    "nested_rings": 1.0,
    "random_spread": 1.0,
    "discrete_obstacles": 1.0,
}

EXTRA_CONDITIONS = [
    # {"name": "slope_10deg_downhill_friction_0.3", "terrain": "slope", "slope_deg": 10,
    #  "direction": "downhill", "command": {"vx": 0.3}, "hold_heading": True, "friction": 0.3},
]
ONLY = None                            # e.g. ["flat", "slope_10deg_downhill"] to run a subset
PUSHES = False
SAVE_STEPS = True
SAVE_VIDEO = True                      # video of one evaluated episode per condition (a few MB each)
FLOOR_TEXTURE = True                   # 20 cm checkerboard on the terrain, visual only (no physical effect)
SEED = 0
MAX_ENVS = 256                         # copies simulated in parallel; 256 is safe on a T4
DRIVE_DIR = "/content/drive/MyDrive/microduck_thesis/eval"
SUITE_NAME = f"{POLICY}_seed{SEED}_v2"   # v2: pyramid slopes and heading hold (older suites used the ramp)
# =====================================================

CONDITIONS = [
    {"name": "flat"},
    {"name": "flat_forward", "heading_deg": 0, "command": {"vx": FORWARD_SPEED}, "hold_heading": HOLD_HEADING},
]
for deg in SLOPES_DEG:
    for direction in SLOPE_DIRECTIONS:
        CONDITIONS.append({"name": f"slope_{deg}deg_{direction}", "terrain": "slope", "slope_deg": deg,
                           "direction": direction, "tile_size_m": SLOPE_TILE_SIZE,
                           "command": {"vx": FORWARD_SPEED}, "hold_heading": HOLD_HEADING})
for mu in FRICTIONS:
    CONDITIONS.append({"name": f"friction_{mu}", "friction": mu})
for name, (mass, offset) in PAYLOADS.items():
    CONDITIONS.append({"name": name, "payload": {"mass_kg": mass, "offset_m": list(offset)}})
for terrain, difficulty in TERRAINS.items():
    CONDITIONS.append({"name": f"{terrain}_d{difficulty}", "terrain": terrain, "difficulty": difficulty})
CONDITIONS += EXTRA_CONDITIONS

names = [c["name"] for c in CONDITIONS]
assert len(names) == len(set(names)), "condition names must be unique"
selected = [c for c in CONDITIONS if not ONLY or c["name"] in ONLY]
print(f"suite '{SUITE_NAME}': {len(selected)} conditions x {EPISODES_PER_CONDITION} episodes "
      f"= {len(selected) * EPISODES_PER_CONDITION} episodes")
for c in selected:
    print(" -", c["name"])
"""),
    md(r"""
## 8. Run the suite

Runs every selected condition in sequence and then shows the results table. The first condition also compiles the MuJoCo Warp kernels (a few minutes); the following ones reuse them. Conditions already saved in the suite folder are skipped: **after changing the script, delete the old condition folders** (or change `SUITE_NAME`), otherwise the old results are kept.

How to read the table:
- `truncated` must be **0** and `mean_duration_s` close to `EPISODE_LENGTH_S` (minus the falls): episodes that end early without a fall mean the condition was not simulated correctly, and its numbers must not be used;
- `max_tilt_deg` is the mean over episodes of the largest trunk tilt from vertical: it shows how close the robot came to falling, also when it did not fall;
- velocity, all in the robot frame:
  - `vel_error_m_s` is the **instantaneous** error |velocity − command|. The trunk of a small biped swings forward/back and sideways within every step, so this error stays around 0.2 m/s even when the average speed is tracked well: it mostly measures the gait oscillation, not the tracking;
  - `vel_error_1s_m_s` compares velocity and command both **averaged over about 1 s** (after a 1 s warm-up): this is the tracking error to report;
  - `cmd_fwd_m_s` and `fwd_vel_m_s` are the mean commanded and actual forward velocity: with a constant command (`flat_forward`, slopes) they should be close;
  - `speed_m_s` is the distance actually covered divided by the episode duration;
- heading:
  - `yaw_rate_abs` is the mean over episodes of |mean yaw rate| (rad/s), `heading_change_deg` the mean |heading change| over the episode, `left/right` how many episodes turn left / right (|mean yaw rate| > 0.02 rad/s);
  - with heading hold (slopes, `flat_forward`) they show how well the heading is kept; with random commands they include the commanded turns.
"""),
    code(r"""
import json
import pandas as pd

suite_dir = os.path.join(DRIVE_DIR, SUITE_NAME)
os.makedirs(suite_dir, exist_ok=True)

for i, cond in enumerate(selected, 1):
    if os.path.exists(os.path.join(suite_dir, cond["name"], "summary.json")):
        print(f"[{i}/{len(selected)}] {cond['name']}: already done, skipped")
        continue
    conf = {
        "run_name": cond["name"], "policy": POLICY_PATH, "policy_name": POLICY, "condition": cond,
        "num_episodes": EPISODES_PER_CONDITION, "episode_length_s": EPISODE_LENGTH_S,
        "pushes": PUSHES, "save_steps": SAVE_STEPS, "save_video": SAVE_VIDEO, "floor_texture": FLOOR_TEXTURE,
        "seed": SEED, "max_envs": MAX_ENVS, "drive_dir": suite_dir,
    }
    with open("eval_config.json", "w") as f:
        json.dump(conf, f, indent=1)
    print(f"\n[{i}/{len(selected)}] {cond['name']}")
    !uv run --with "imageio[ffmpeg]" python eval_scenario.py eval_config.json

# ---- results table ----
results, missing, warnings = [], [], {}
for cond in CONDITIONS:
    path = os.path.join(suite_dir, cond["name"], "summary.json")
    if not os.path.exists(path):
        missing.append(cond["name"])
        continue
    with open(path) as f:
        run = json.load(f)
    s = run["summary"]
    mean = lambda key: s[key]["mean"] if s.get(key) else None   # noqa: E731  (older runs lack some keys)
    low, high = s["falls_per_100_ci95"]
    results.append({
        "condition": cond["name"], "episodes": s["episodes"], "falls_per_100": s["falls_per_100"],
        "ci95_low": low, "ci95_high": high,
        "truncated": s["truncated"],              # must be 0: otherwise the condition is not valid
        "mean_duration_s": mean("duration_s"),
        "max_tilt_deg": mean("max_tilt_deg"),
        "vel_error_m_s": mean("vel_error_m_s"),   # instantaneous, includes the gait oscillation
        "vel_error_1s_m_s": mean("vel_error_1s_m_s"),
        "cmd_fwd_m_s": mean("cmd_forward_vel_m_s"),
        "fwd_vel_m_s": mean("forward_vel_m_s"),
        "speed_m_s": mean("mean_speed_m_s"),
        "yaw_rate_abs": mean("abs_yaw_rate_rad_s"),
        "heading_change_deg": mean("abs_heading_change_deg"),
        "left/right": f"{s.get('episodes_turning_left', '-')}/{s.get('episodes_turning_right', '-')}",
        "power_W": mean("mean_power_W"),
    })
    if run["meta"]["warnings"]:
        warnings[cond["name"]] = run["meta"]["warnings"]

with open(os.path.join(suite_dir, "suite_summary.json"), "w") as f:
    json.dump({"policy": POLICY, "policy_path": POLICY_PATH, "seed": SEED,
               "episodes_per_condition": EPISODES_PER_CONDITION, "episode_length_s": EPISODE_LENGTH_S,
               "conditions": CONDITIONS, "results": results}, f, indent=1)

display(pd.DataFrame(results).round(3))
if missing:
    print("not available (failed or not run):", ", ".join(missing))
for name, w in warnings.items():
    print(f"WARNINGS in {name}:", *w, sep="\n - ")
"""),
    md(r"""
## 9. Compare two policies

Run the suite once per policy (change `POLICY` in cell 5, re-run cells 5, 7 and 8), then compare the two suites. `A worse` marks the conditions where the 95% intervals of the fall rate do not overlap and A falls more: a conservative test, so a missing mark does not mean "no difference".
"""),
    code(r"""
SUITE_A = "alpha_walking_seed0_v2"
SUITE_B = "rough_walk_g_seed0_v2"

def load_suite(name):
    with open(os.path.join(DRIVE_DIR, name, "suite_summary.json")) as f:
        return pd.DataFrame(json.load(f)["results"]).set_index("condition")

a, b = load_suite(SUITE_A), load_suite(SUITE_B)
cols = [c for c in ["falls_per_100", "ci95_low", "ci95_high", "vel_error_1s_m_s", "cmd_fwd_m_s",
                     "fwd_vel_m_s", "yaw_rate_abs", "power_W"] if c in a.columns and c in b.columns]
comparison = a[cols].join(b[cols], lsuffix=" A", rsuffix=" B", how="outer")
comparison["falls diff (A-B)"] = comparison["falls_per_100 A"] - comparison["falls_per_100 B"]
comparison["A worse"] = comparison["ci95_low A"] > comparison["ci95_high B"]
print(f"A = {SUITE_A}   B = {SUITE_B}")
display(comparison.round(2))
"""),
    md(r"""
## 10. Reading the data back

Example: one condition's episodes in a table and its steps as NumPy arrays. Works in a fresh session too: mount Drive and set the suite and condition names.
"""),
    code(r"""
import numpy as np

run_dir = os.path.join(DRIVE_DIR, SUITE_NAME, "flat")   # any condition folder

with open(os.path.join(run_dir, "summary.json")) as f:
    run = json.load(f)
episodes = pd.DataFrame(run["episodes"])
display(episodes.head())

if os.path.exists(os.path.join(run_dir, "steps.npz")):
    steps = np.load(os.path.join(run_dir, "steps.npz"))
    for name in steps.files:
        print(f"{name:18s} {str(steps[name].shape):16s} {steps[name].dtype}")
    # e.g. second hidden layer activations in the steps that precede a fall
    h2_before_fall = steps["h2"][steps["falls_within_2s"]]
    print("steps less than 2 s before a fall:", len(h2_before_fall))
"""),
    md(r"""
### Velocity check

Uses `steps.npz` (also from runs made before the new velocity metrics) to see whether the robot walks at the commanded speed. With a constant command (`flat_forward`, slopes):
- **mean actual vx close to the command** and a large standard deviation → the robot tracks the speed and the instantaneous error is the gait oscillation;
- **mean actual vx close to 0** → the robot steps in place or barely moves: the policy does not follow the command (or the command is not the one we think).
"""),
    code(r"""
CHECK_CONDITION = "flat_forward"

run_dir = os.path.join(DRIVE_DIR, SUITE_NAME, CHECK_CONDITION)
steps = np.load(os.path.join(run_dir, "steps.npz"))
with open(os.path.join(run_dir, "summary.json")) as f:
    layout = json.load(f)["meta"]["obs_layout_actor"]
cmd = steps["obs_actor"][:, slice(*layout.get("command", [48, 51]))]
vel = steps["lin_vel_b"]
settled = steps["t_s"] >= 1.0   # skip the first second after the spawn

print(f"commanded vx: mean {cmd[settled, 0].mean():+.3f} m/s")
print(f"actual    vx: mean {vel[settled, 0].mean():+.3f} m/s   std {vel[settled, 0].std():.3f}")
print(f"actual    vy: mean {vel[settled, 1].mean():+.3f} m/s   std {vel[settled, 1].std():.3f}")
inst = np.linalg.norm(vel[settled, :2] - cmd[settled, :2], axis=1).mean()
print(f"instantaneous error |v - cmd|: {inst:.3f} m/s")

# per-episode mean velocity: averaging over the episode removes the within-step oscillation
episode = steps["episode"][settled]
count = np.bincount(episode)
ok = count > 0
mean_vx = np.bincount(episode, vel[settled, 0]) / np.maximum(count, 1)
mean_cmd = np.bincount(episode, cmd[settled, 0]) / np.maximum(count, 1)
print(f"per-episode mean vx: {mean_vx[ok].mean():+.3f} m/s (min {mean_vx[ok].min():+.3f}, "
      f"max {mean_vx[ok].max():+.3f}) vs command {mean_cmd[ok].mean():+.3f} m/s")
"""),
    md(r"""
## 11. Watching the videos

Shows the saved episode video of every condition in the suite (or only those in `SHOW`), with its outcome.
"""),
    code(r"""
from base64 import b64encode
from IPython.display import HTML

SHOW = None   # e.g. ["flat", "slope_10deg_downhill"]; None = every condition with a video

for cond in CONDITIONS:
    if SHOW and cond["name"] not in SHOW:
        continue
    run_dir = os.path.join(DRIVE_DIR, SUITE_NAME, cond["name"])
    video_path = os.path.join(run_dir, "episode_video.mp4")
    if not os.path.exists(video_path):
        continue
    with open(os.path.join(run_dir, "summary.json")) as f:
        info = json.load(f)["meta"].get("video") or {}
    with open(video_path, "rb") as f:
        data = b64encode(f.read()).decode()
    display(HTML(f'<p><b>{cond["name"]}</b>: episode {info.get("episode_id")}, {info.get("outcome")}, '
                 f'{info.get("duration_s", 0):.1f} s</p>'
                 f'<video width=480 controls src="data:video/mp4;base64,{data}"></video>'))
"""),
    md(r"""
## Troubleshooting

| Message | Likely cause | What to do |
|---|---|---|
| `[WARNING] ... using mjlab defaults` | The Microduck-scaled parameter names do not exist for that terrain type | The message lists the available fields: pass them in `terrain_params` |
| `The PyTorch policy does not match the ONNX one` | Wrong ONNX → PyTorch conversion | Do not use the data; paste the error |
| `unsupported ONNX op` | The graph contains an unexpected operation | Paste the error: it gets added to `load_policy` |
| `expected 2 foot collision geoms` | Foot geoms not found in the scene | Paste the error with the printed names |
| Uphill slopes: falls in the first second (`time_to_fall_s` ~0-1 s), or the video shows the robot dropping onto / sinking into the face | Spawn height on the pyramid face not matching the surface | Check the `[slope]` line printed by the script and `terrain_params_used.spawn` in `summary.json`; paste them |
| `AttributeError` on `heading_command` or `ranges.heading` | Different heading-controller fields in this mjlab version | Paste the error; meanwhile set `HOLD_HEADING = False` |
| `trunk_base body not found` | The trunk has a different name in the scene | Paste the error |
| `foot_friction` / `terrain_friction` in `summary.json` differ from the condition | The friction was overwritten after the scene was built | Do not use that condition; paste the values |
| `AttributeError` on `rel_turn_in_place_envs` or `ranges` | Different field names in the command | Paste the error |
| `WARNINGS: ... not available` | A scenario field was not read | The walking data are still valid; that field is empty in the JSON |
| `video disabled, the renderer could not start` | EGL not available in the evaluation process | Re-run cell 3 before cell 8; the evaluation data are still valid |
| No `episode_video.mp4` for some conditions | Those conditions were run before the video was added, and cell 8 skips them | Delete their folders on Drive (or change `SUITE_NAME`) and re-run cell 8 |
| CUDA out of memory | Too many parallel copies | Lower `MAX_ENVS` |
| Many falls even on `flat` | Policy and observations not aligned | Check that you use a 61-observation walking policy |
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("written", OUT, len(cells), "cells")
