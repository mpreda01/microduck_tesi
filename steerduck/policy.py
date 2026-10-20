"""The Controller interface, the ONNX controller, the policy as a PyTorch module, and the registry
(configs/policies.yaml): build_controller() is the only place where controllers are assembled."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from steerduck.config import CONFIGS, MODELS, load_yaml

N_OBS = 61


class Controller:
    """Everything evaluated in the thesis: 61 observations in, 14 motor offsets out."""

    name = ""

    def reset(self):
        """Clear the internal state at the start of an episode."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def activations(self) -> dict:
        """Hidden-layer activations of the last act() ({h1, h2, h3}), empty if not recorded."""
        return {}

    def episode_diagnostics(self) -> dict:
        """Per-episode checks, added to the episode's results."""
        return {}


class OnnxController(Controller):
    """An ONNX policy run with onnxruntime. With record_activations, a PyTorch copy of the same graph runs
    alongside only to READ the activations: the actions still come from onnxruntime, and the largest
    difference between the two outputs in each episode is reported as a check."""

    def __init__(self, name: str, path, record_activations: bool = False):
        import onnxruntime as ort
        self.name = name
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        if self.sess.get_inputs()[0].shape[-1] != N_OBS:
            raise ValueError(f"{path}: the policy does not take {N_OBS} observations")
        self.torch_model, self.latest, self.max_torch_diff = None, {}, 0.0
        if record_activations:
            import torch
            torch.set_num_threads(1)   # one process per (controller, situation): no thread oversubscription
            self.torch_model = onnx_to_torch(path)
            for k, layer in enumerate(hidden_layers(self.torch_model)):
                layer.register_forward_hook(self._keep(f"h{k + 1}"))

    def _keep(self, name):
        """Forward hook that keeps the layer's output (first item of the batch) as latest[name]."""
        def hook(module, inputs, output):
            self.latest[name] = output[0].numpy().copy()
        return hook

    def reset(self):
        self.max_torch_diff = 0.0

    def act(self, obs):
        action = self.sess.run(None, {self.input: obs[None]})[0][0].astype(np.float32)
        if self.torch_model is not None:
            import torch
            with torch.no_grad():
                y = self.torch_model(torch.from_numpy(obs[None]))[0].numpy()
            self.max_torch_diff = max(self.max_torch_diff, float(np.abs(y - action).max()))
        return action

    def activations(self):
        return dict(self.latest)

    def episode_diagnostics(self):
        return {"torch_onnx_max_diff": self.max_torch_diff} if self.torch_model is not None else {}


# ---------------------------------------------------------------------------
# The policy in PyTorch (plan section 4.5)
# ---------------------------------------------------------------------------
def onnx_to_torch(path):
    """The ONNX graph of a Microduck policy as an nn.Sequential: MLP 61 -> 512 -> 256 -> 128 -> 14 with ELU,
    possibly with rsl_rl's input normalization before and a clip after. Nodes on constants only are
    folded; Gemm/MatMul -> Linear; elementwise ops with a constant -> Elementwise; Clip -> Hardtanh."""
    import onnx
    import torch
    from onnx import numpy_helper

    ops = {"Sub": torch.sub, "Add": torch.add, "Mul": torch.mul, "Div": torch.div}
    fold = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide, "Sqrt": np.sqrt,
            "Identity": lambda x: x, "Cast": lambda x: x}

    class Elementwise(torch.nn.Module):
        def __init__(self, op, c):
            super().__init__()
            self.op = op
            self.register_buffer("c", torch.as_tensor(np.asarray(c, dtype=np.float32)))

        def forward(self, x):
            return ops[self.op](x, self.c)

    graph = onnx.load(str(path)).graph
    const = {i.name: numpy_helper.to_array(i).copy() for i in graph.initializer}
    mods = []
    for n in graph.node:
        at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        op, ins = n.op_type, list(n.input)
        if op == "Constant":
            const[n.output[0]] = numpy_helper.to_array(at["value"])
        elif all(i in const for i in ins if i):   # node on constants only: fold it now
            if op not in fold:
                raise ValueError(f"unsupported ONNX op on constants: {op}")
            const[n.output[0]] = fold[op](*[const[i] for i in ins if i])
        elif op in ("Gemm", "MatMul"):
            w = const[ins[1]] if (op == "Gemm" and at.get("transB", 0)) else const[ins[1]].T
            lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=op == "Gemm" and len(ins) > 2 and ins[2] in const)
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)))
                if lin.bias is not None:
                    lin.bias.copy_(torch.from_numpy(const[ins[2]].astype(np.float32).reshape(-1)))
            mods.append(lin)
        elif op in ops:
            c = const[ins[1]]
            if op == "Add" and mods and isinstance(mods[-1], torch.nn.Linear) and mods[-1].bias is None:
                mods[-1].bias = torch.nn.Parameter(torch.from_numpy(c.astype(np.float32).reshape(-1)))
            else:
                mods.append(Elementwise(op, c))
        elif op == "Elu":
            mods.append(torch.nn.ELU(alpha=at.get("alpha", 1.0)))
        elif op == "Clip":
            lo = float(const[ins[1]]) if len(ins) > 1 and ins[1] else at.get("min", -np.inf)
            hi = float(const[ins[2]]) if len(ins) > 2 and ins[2] else at.get("max", np.inf)
            mods.append(torch.nn.Hardtanh(lo, hi))   # = clamp(x, lo, hi)
        elif op not in ("Identity", "Cast", "Flatten", "Reshape", "Squeeze", "Unsqueeze"):
            raise ValueError(f"unsupported ONNX op: {op}")
    return torch.nn.Sequential(*mods).eval()


def hidden_layers(model) -> list:
    """The three ELUs of the policy: their outputs are the activations h1, h2, h3."""
    import torch
    elus = [m for m in model if isinstance(m, torch.nn.ELU)]
    if len(elus) != 3:
        raise ValueError(f"expected 3 hidden layers (ELU), found {len(elus)}")
    return elus


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def resolve(name: str) -> dict:
    """-> {name, type, path}, downloading the policy to models/<name>/ if needed. Called in the main
    process; the result is passed to the worker processes."""
    registry = load_yaml(CONFIGS / "policies.yaml")
    if name not in registry:
        raise KeyError(f"controller {name!r} not in configs/policies.yaml; known: {', '.join(registry)}")
    entry = registry[name]
    if "path" in entry:
        path = MODELS / entry["path"]
    else:
        repo, filename = entry["hf"]
        path = MODELS / name / filename
        if not path.exists():
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo, filename, local_dir=str(MODELS / name))
    if not Path(path).exists():
        raise FileNotFoundError(f"controller {name!r}: {path} not found")
    return {"name": name, "type": entry["type"], "path": str(path)}


def build_controller(spec: dict, record_activations: bool = False) -> Controller:
    if spec["type"] == "onnx":
        return OnnxController(spec["name"], spec["path"], record_activations)
    raise ValueError(f"unknown controller type {spec['type']!r}")
