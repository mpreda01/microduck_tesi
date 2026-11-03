"""The SAE and the linear probes on the activations of a controller (plan section 3.6, phase 3 step 1; glossary and
section 6.3 for the probes), trained on the per-step data of an evaluation run.

torch is imported at the top, unlike the rest of the library: every SAE function needs it, and this module is
imported only where the SAE is used.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from steerduck.report import recorded_steps

log = logging.getLogger("steerduck")


def load_dataset(run_dir, controller: str, layers: list, quantities: list, stride: int, test_every: int):
    """Activations of `layers` and ground truth `quantities` recorded by `controller` in an evaluation run, every
    `stride`-th step. Train and test steps never come from the same episode: test = the episodes whose number is a
    multiple of test_every. -> ({layer: float32 array}, ground truth DataFrame, boolean test mask, situation of each
    step)."""
    acts, truth, test, situations = {layer: [] for layer in layers}, [], [], []
    for s, d in recorded_steps(run_dir, controller, fields=[*layers, *quantities, "episode"]):
        situations.append(np.full(len(d["episode"][::stride]), s["name"]))
        for layer in layers:
            acts[layer].append(d[layer][::stride].astype(np.float32))
        columns = {}
        for q in quantities:   # one column per component: gravity_b -> gravity_b_0, gravity_b_1, gravity_b_2
            values = d[q][::stride].astype(np.float32).reshape(len(d[q][::stride]), -1)
            columns |= {q if values.shape[1] == 1 else f"{q}_{i}": values[:, i] for i in range(values.shape[1])}
        truth.append(pd.DataFrame(columns))
        test.append(d["episode"][::stride] % test_every == 0)
    if not truth:
        raise ValueError(f"{run_dir.name}: no per-step data of {controller}")
    return ({layer: np.concatenate(a) for layer, a in acts.items()}, pd.concat(truth, ignore_index=True),
            np.concatenate(test), np.concatenate(situations))


def probe_r2(acts: dict, truth: pd.DataFrame, test: np.ndarray) -> pd.DataFrame:
    """Linear probes: for each layer, a linear regression from its activations to every ground truth column, fitted
    on the train episodes; R² on the test episodes says how readable each quantity is in each layer (1 = fully)."""
    rows = []
    for layer, x in acts.items():
        x = np.hstack([x, np.ones((len(x), 1), np.float32)])   # intercept
        w = np.linalg.lstsq(x[~test], truth.values[~test], rcond=None)[0]
        y, pred = truth.values[test], x[test] @ w
        r2 = 1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
        rows += [{"layer": layer, "quantity": q, "r2": r} for q, r in zip(truth.columns, r2)]
    return pd.DataFrame(rows)


class SAE(torch.nn.Module):
    """TopK sparse autoencoder: an activation h of `layer` is written as a sum of at most k of the n dictionary
    elements, the columns of the decoder (kept at unit norm): z = TopK(ReLU(W_enc (h - b_dec) + b_enc)),
    h ≈ W_dec z + b_dec."""

    def __init__(self, layer: str, d: int, n: int, k: int):
        super().__init__()
        self.layer, self.k = layer, k
        self.enc, self.dec = torch.nn.Linear(d, n), torch.nn.Linear(n, d)
        with torch.no_grad():
            self.dec.weight /= self.dec.weight.norm(dim=0)
            self.enc.weight.copy_(self.dec.weight.T)
            self.enc.bias.zero_()

    def encode(self, h):
        z = torch.relu(self.enc(h - self.dec.bias))
        top = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter(-1, top.indices, top.values)

    def forward(self, h):
        return self.dec(self.encode(h))


def encode(sae: SAE, x: np.ndarray, batch: int = 8192):
    """Dictionary weights z and reconstruction of every row of x, in batches -> (z, reconstruction) as numpy."""
    device, z, recon = next(sae.parameters()).device, [], []
    with torch.no_grad():
        for rows in np.array_split(x, max(1, len(x) // batch)):
            zb = sae.encode(torch.from_numpy(rows).to(device))
            z.append(zb.cpu().numpy())
            recon.append(sae.dec(zb).cpu().numpy())
    return np.concatenate(z), np.concatenate(recon)


def fvu(x: np.ndarray, recon: np.ndarray) -> float:
    """Fraction of the variance of x that the reconstruction misses (0 = perfect)."""
    return float(((x - recon) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())


def error_by_situation(x: np.ndarray, recon: np.ndarray, situations: np.ndarray) -> pd.DataFrame:
    """Reconstruction error of each situation, as its mean squared error over the mean variance of x on all the steps:
    comparable across situations, and its mean over the steps is the overall FVU. Shows where the SAE fails."""
    sq = ((x - recon) ** 2).mean(1) / x.var(0).mean()
    return (pd.DataFrame({"situation": situations, "error": sq}).groupby("situation", sort=False)
            .agg(steps=("error", "size"), error=("error", "mean")).reset_index())


def train_sae(x: np.ndarray, test: np.ndarray, layer: str, conf: dict) -> SAE:
    """A TopK SAE on the train rows of x (the activations of `layer`), with the hyperparameters of the sae section of
    configs/training.yaml; the fraction of unexplained variance on the test rows is logged after every epoch."""
    torch.manual_seed(conf["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = torch.from_numpy(x[~test])
    sae = SAE(layer, x.shape[1], conf["dictionary_size"], conf["k"]).to(device)
    with torch.no_grad():
        sae.dec.bias.copy_(train.mean(0))
    optimizer = torch.optim.Adam(sae.parameters(), lr=conf["learning_rate"])
    for epoch in range(conf["epochs"]):
        for batch in torch.randperm(len(train)).split(conf["batch_size"]):
            h = train[batch].to(device)
            loss = ((sae(h) - h) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                sae.dec.weight /= sae.dec.weight.norm(dim=0)
        log.info(f"{layer} epoch {epoch + 1}/{conf['epochs']}: test FVU {fvu(x[test], encode(sae, x[test])[1]):.4f}")
    return sae.cpu().eval()


def name_features(z: np.ndarray, truth: pd.DataFrame) -> pd.DataFrame:
    """Each dictionary element named after the ground truth column its weight correlates with most (plan section
    3.6): how often it is active, that column and Pearson's r. Elements never active get no name."""
    with np.errstate(invalid="ignore", divide="ignore"):
        zs = (z - z.mean(0)) / z.std(0)
        ts = ((truth - truth.mean()) / truth.std()).values
        r = np.nan_to_num(zs.T @ ts / len(z))
    best = np.abs(r).argmax(1)
    active = (z > 0).mean(0)
    return pd.DataFrame({"feature": np.arange(len(best)), "active_pct": 100 * active,
                         "quantity": np.where(active > 0, truth.columns[best], None),
                         "r": r[np.arange(len(best)), best]})


def save_sae(sae: SAE, path):
    torch.save({"layer": sae.layer, "d": sae.enc.in_features, "n": sae.enc.out_features, "k": sae.k,
                "state_dict": sae.state_dict()}, path)


def load_sae(path) -> SAE:
    saved = torch.load(path, map_location="cpu")
    sae = SAE(saved["layer"], saved["d"], saved["n"], saved["k"])
    sae.load_state_dict(saved["state_dict"])
    return sae.eval()
