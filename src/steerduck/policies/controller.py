"""The Controller interface (REPO_STRUCTURE.md, section 1) and the controller of an ONNX policy."""
from __future__ import annotations

import numpy as np

N_OBS = 61


class Controller:
    """Everything evaluated in the thesis: 61 observations in, 14 motor offsets out."""

    name = ""

    def reset(self):
        """Clear the internal state at the start of an episode."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def activations(self) -> dict | None:
        """Hidden-layer activations of the last act() (h1, h2, h3), or None if not recorded."""
        return None

    def episode_diagnostics(self) -> dict:
        """Per-episode checks added to the episode's results."""
        return {}


class OnnxController(Controller):
    """An ONNX policy run with onnxruntime. With record_activations a PyTorch copy of the same graph
    (policies/mlp.py) runs alongside to READ the activations: the actions still come from onnxruntime,
    and the largest difference between the two outputs in each episode is reported as a check."""

    def __init__(self, name: str, path, record_activations: bool = False):
        import onnxruntime as ort
        self.name = name
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        n = self.sess.get_inputs()[0].shape[-1]
        if n != N_OBS:
            raise ValueError(f"{path}: the policy takes {n} observations, the simulator builds the {N_OBS}-D layout")
        self.torch_model = self.recorder = None
        if record_activations:
            import torch
            from steerduck.policies.mlp import hidden_layers, load_onnx_as_torch
            from steerduck.steering.hooks import ActivationRecorder
            torch.set_num_threads(1)   # one process per (controller, situation): no thread oversubscription
            self.torch_model = load_onnx_as_torch(path)
            self.recorder = ActivationRecorder(hidden_layers(self.torch_model))
        self.max_torch_diff = 0.0

    def reset(self):
        self.max_torch_diff = 0.0

    def act(self, obs):
        action = self.sess.run(None, {self.inp: obs[None]})[0][0].astype(np.float32)
        if self.torch_model is not None:
            import torch
            with torch.no_grad():
                y = self.torch_model(torch.from_numpy(obs[None]))[0].numpy()
            self.max_torch_diff = max(self.max_torch_diff, float(np.abs(y - action).max()))
        return action

    def activations(self):
        return dict(self.recorder.latest) if self.recorder is not None else None

    def episode_diagnostics(self):
        return {"torch_onnx_max_diff": self.max_torch_diff} if self.torch_model is not None else {}
