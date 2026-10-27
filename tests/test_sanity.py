"""The checks without which the results would be wrong (REPO_STRUCTURE.md, section 6)."""
import numpy as np
import pytest

from steerduck.policy import actor_onnx_difference, hidden_layers, onnx_to_torch, resolve, resolve_checkpoint


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


def test_checkpoint_actor_equivalence():
    """The actor of alpha_walking's training checkpoint, the start of every fine-tuning, gives the same
    actions as the ONNX evaluated in phase 1, normalization included."""
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    path = resolve("alpha_walking")["path"]
    obs = TensorDict({"actor": torch.zeros(1, 61)}, batch_size=[1])
    actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 14, (512, 256, 128), "elu", obs_normalization=True,
                     distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"})
    checkpoint = torch.load(resolve_checkpoint("alpha_walking"), map_location="cpu", weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    assert actor_onnx_difference(actor, path) < 1e-5
