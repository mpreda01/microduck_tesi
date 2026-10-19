"""The PyTorch copy of a policy matches onnxruntime (plan section 4.5): without this every activation
recorded or steered later would be unreliable."""
import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")

from steerduck.policies.mlp import hidden_layers, load_onnx_as_torch  # noqa: E402
from steerduck.policies.registry import resolve  # noqa: E402


@pytest.mark.parametrize("name", ["alpha_walking", "rough_walk_g"])
def test_torch_matches_onnx(name):
    path = resolve(name)["path"]
    model = load_onnx_as_torch(path)
    assert len(hidden_layers(model)) == 3
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(10_000, 61)).astype(np.float32)
    inp = sess.get_inputs()[0].name
    onnx_out = np.concatenate([sess.run(None, {inp: row[None]})[0] for row in x])
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()
    assert np.abs(torch_out - onnx_out).max() < 1e-5
