"""PyTorch forward hooks on the hidden layers of a policy (plan sections 4.9-4.10)."""
from __future__ import annotations

import numpy as np


class ActivationRecorder:
    """Keeps the latest output of each given layer, as numpy arrays of the first item of the batch.
    Layers are named h1, h2, h3... in order."""

    def __init__(self, layers):
        self.latest: dict[str, np.ndarray] = {}
        self.handles = [layer.register_forward_hook(self._hook(f"h{k + 1}")) for k, layer in enumerate(layers)]

    def _hook(self, name):
        def hook(module, inputs, output):
            self.latest[name] = output[0].detach().cpu().numpy().copy()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
