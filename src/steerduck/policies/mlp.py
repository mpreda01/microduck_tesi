"""The walking policy as a PyTorch nn.Sequential, rebuilt from its ONNX graph (plan section 4.5).

The graph of the Microduck policies is an MLP 61 -> 512 -> 256 -> 128 -> 14 with ELU, possibly preceded
by the input normalization of rsl_rl and followed by a clip: constants are folded, Gemm/MatMul become
nn.Linear, elementwise ops on constants become `Elementwise`, Clip becomes Hardtanh.
"""
from __future__ import annotations

import numpy as np
import torch

_ELEMENTWISE = {"Sub": torch.sub, "Add": torch.add, "Mul": torch.mul, "Div": torch.div}
_FOLDING = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}
_SKIPPED = ("Identity", "Cast", "Flatten", "Reshape", "Squeeze", "Unsqueeze")


class Elementwise(torch.nn.Module):
    """x (op) constant, e.g. the subtraction of the observation mean."""

    def __init__(self, op: str, c):
        super().__init__()
        self.op = op
        self.register_buffer("c", torch.as_tensor(np.asarray(c, dtype=np.float32)))

    def forward(self, x):
        return _ELEMENTWISE[self.op](x, self.c)


def load_onnx_as_torch(path) -> torch.nn.Sequential:
    import onnx
    from onnx import numpy_helper

    g = onnx.load(str(path)).graph
    const = {i.name: numpy_helper.to_array(i).copy() for i in g.initializer}
    mods = []
    for n in g.node:
        at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        op, ins = n.op_type, list(n.input)
        if op == "Constant":
            const[n.output[0]] = numpy_helper.to_array(at["value"])
            continue
        if all(i in const for i in ins if i):   # node on constants only: fold it now
            v = [const[i] for i in ins if i]
            if op in _FOLDING:
                const[n.output[0]] = _FOLDING[op](*v)
            elif op == "Sqrt":
                const[n.output[0]] = np.sqrt(v[0])
            elif op in ("Identity", "Cast"):
                const[n.output[0]] = v[0]
            else:
                raise ValueError(f"unsupported ONNX op on constants: {op}")
            continue
        if op in ("Gemm", "MatMul"):
            w = const[ins[1]] if (op == "Gemm" and at.get("transB", 0)) else const[ins[1]].T
            lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=op == "Gemm" and len(ins) > 2 and ins[2] in const)
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)))
                if lin.bias is not None:
                    lin.bias.copy_(torch.from_numpy(const[ins[2]].astype(np.float32).reshape(-1)))
            mods.append(lin)
        elif op in _ELEMENTWISE:
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
        elif op in _SKIPPED:
            continue
        else:
            raise ValueError(f"unsupported ONNX op: {op}")
    return torch.nn.Sequential(*mods).eval()


def hidden_layers(model: torch.nn.Sequential, expected: int = 3) -> list:
    """The ELU modules of the policy, in order: their outputs are the activations h1, h2, h3."""
    elus = [m for m in model if isinstance(m, torch.nn.ELU)]
    if len(elus) != expected:
        raise ValueError(f"expected {expected} hidden layers (ELU), found {len(elus)}")
    return elus
