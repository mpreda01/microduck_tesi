"""Forward hooks that change the activations of the policy (plan sections 3.1 and 4.10). For now the controller with
one hidden layer replaced by a function of itself, used to check that the policy still walks with the SAE's
reconstruction in place of the activation (plan section 3.6, phase 3 step 1)."""
from __future__ import annotations

import numpy as np

from steerduck.policy import Controller, hidden_layers, onnx_to_torch


class HookedController(Controller):
    """The policy run in PyTorch, with the activation of `layer` (h1, h2 or h3) replaced by edit(activation)."""

    def __init__(self, name: str, path, layer: str, edit):
        import torch
        torch.set_num_threads(1)   # one process per (controller, situation): no thread oversubscription
        self.name, self.model = name, onnx_to_torch(path)
        hidden_layers(self.model)[int(layer[1]) - 1].register_forward_hook(lambda module, inputs, out: edit(out))

    def act(self, obs):
        import torch
        with torch.no_grad():
            return self.model(torch.from_numpy(obs[None]))[0].numpy().astype(np.float32)
