"""The DAgger loop of plan section 3.2 (steps 2-4): a constant steering vector, added to the activations of the
frozen base policy, is fitted to imitate a specialist, first on the states the specialist visits, then also on those
the steered policy visits itself, labelled by the specialist. The rollouts are evaluation runs of their own
(rollout.evaluate_run), with the protocol of configs/eval.yaml."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from steerduck.config import setup_logging
from steerduck.policy import onnx_to_torch, relabel
from steerduck.report import add_results, recorded_steps
from steerduck.rollout import evaluate_run
from steerduck.steering import LAYER_SIZES, const_edits, hook_layers

log = logging.getLogger("steerduck")


def rollout(ev: dict, spec: dict, situations: list, run_dir):
    """Evaluate one controller (a run of its own) -> (episodes, its observations, its actions)."""
    eval_dir = evaluate_run(ev, [spec], situations, f"{spec['name']}_dagger")
    setup_logging(run_dir / "log.txt")   # back to the distillation's log
    steps = [d for _, d in recorded_steps(eval_dir, spec["name"], fields=("obs", "action"))]
    episodes = pd.read_parquet(eval_dir / "episodes.parquet")
    return episodes, np.concatenate([d["obs"] for d in steps]), np.concatenate([d["action"] for d in steps])


def fit(model, vectors: dict, obs: np.ndarray, target: np.ndarray, conf: dict) -> float:
    """Adam on the vectors only: squared difference between the steered actions and the specialist's (plan section
    3.2, step 3) -> mean squared error over the last epoch."""
    x, y = torch.from_numpy(obs), torch.from_numpy(target)
    optimizer = torch.optim.Adam(vectors.values(), lr=conf["learning_rate"])
    for _ in range(conf["epochs"]):
        total = 0.0
        for batch in torch.randperm(len(x)).split(conf["batch_size"]):
            loss = ((model(x[batch]) - y[batch]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
    return total / len(x)


def distill(conf: dict, ev: dict, base: dict, specialist: dict, layer: str, situations: list, name: str, run_dir):
    """DAgger for the vector of `layer` (h1, h2, h3 or all). Each iteration fits the vectors on the dataset, saves them
    as adapter_it<k>.pt and evaluates the steered policy. The best iteration has most successes, then fewest falls
    (the main metric: uphill alpha_walking stands still rather than falling); the loop stops after `patience`
    iterations without a better one. -> (one row per iteration, the best iteration)."""
    torch.manual_seed(conf["seed"])
    vectors = {h: torch.zeros(LAYER_SIZES[h], requires_grad=True) for h in (LAYER_SIZES if layer == "all" else [layer])}
    model = hook_layers(onnx_to_torch(base["path"]).requires_grad_(False), const_edits(vectors))
    _, obs, target = rollout(ev, specialist, situations, run_dir)   # step 2: the specialist's own states
    rows, best = [], 0
    for k in range(conf["max_iterations"]):
        mse = fit(model, vectors, obs, target, conf)
        torch.save({h: v.detach().clone() for h, v in vectors.items()}, run_dir / f"adapter_it{k}.pt")
        steered = base | {"name": f"{name}_it{k}", "type": "adapter", "adapter": str(run_dir / f"adapter_it{k}.pt")}
        episodes, new_obs, _ = rollout(ev, steered, situations, run_dir)
        rows.append({"iteration": k, "steps": len(obs), "mse": mse, "episodes": len(episodes),
                     "successes": int((add_results(episodes).result == "success").sum()),
                     "falls": int((episodes.outcome == "fell").sum()), "evaluation": episodes.run_id.iloc[0]})
        log.info(f"DAgger iteration {k}: {len(obs)} steps, mse {mse:.5f}, successes {rows[-1]['successes']}, "
                 f"falls {rows[-1]['falls']} of {len(episodes)}")
        best = min(best, k, key=lambda i: (-rows[i]["successes"], rows[i]["falls"]))
        if k - best >= conf["patience"]:
            break
        obs, target = np.concatenate([obs, new_obs]), np.concatenate([target, relabel(specialist, new_obs)])
    return pd.DataFrame(rows), best
