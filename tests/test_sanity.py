"""The checks without which the results would be wrong (REPO_STRUCTURE.md, section 6)."""
import numpy as np
import pytest

from steerduck.policy import hidden_layers, onnx_to_torch, resolve


@pytest.mark.parametrize("name", ["alpha_walking", "rough_walk_g"])
def test_policy_equivalence(name):
    """The PyTorch copy of the policy gives the same outputs as onnxruntime (plan section 4.5)."""
    import onnxruntime as ort
    import torch

    path = resolve(name)["path"]
    model = onnx_to_torch(path)
    assert len(hidden_layers(model)) == 3
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(10_000, 61)).astype(np.float32)
    expected = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: row[None]})[0] for row in x])
    with torch.no_grad():
        assert np.abs(model(torch.from_numpy(x)).numpy() - expected).max() < 1e-5
