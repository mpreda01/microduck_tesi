"""Forward hooks that change the activations of the policy (plan sections 3.1 and 4.10): the controller with hidden
layers replaced by a function of themselves, used for the SAE reconstruction (plan section 3.6, phase 3 step 1) and
for the constant steering vector (plan section 3.3, variant 1)."""
from __future__ import annotations

import numpy as np

from steerduck.policy import Controller, hidden_layers, onnx_to_torch

LAYER_SIZES = {"h1": 512, "h2": 256, "h3": 128}


def hook_layers(model, edits: dict):
    """Replace the activation of each layer of `edits` (h1, h2, h3) with edits[layer](activation) -> the model."""
    layers = hidden_layers(model)
    for layer, edit in edits.items():
        layers[int(layer[1]) - 1].register_forward_hook(lambda module, inputs, out, edit=edit: edit(out))
    return model


def const_edits(vectors: dict) -> dict:
    """The constant-vector adapter: each activation plus the vector of its layer ({layer: tensor})."""
    return {layer: (lambda h, v=v: h + v) for layer, v in vectors.items()}


class HookedController(Controller):
    """The policy run in PyTorch, with the activation of each layer of `edits` replaced by edits[layer](activation)."""

    def __init__(self, name: str, path, edits: dict):
        import torch
        torch.set_num_threads(1)   # one process per (controller, situation): no thread oversubscription
        self.name, self.model = name, hook_layers(onnx_to_torch(path), edits)

    def act(self, obs):
        import torch
        with torch.no_grad():
            return self.model(torch.from_numpy(obs[None]))[0].numpy().astype(np.float32)
