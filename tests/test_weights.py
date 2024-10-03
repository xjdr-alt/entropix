import pytest
import jax.numpy as jnp
from entropix.weights import load_weights, XfmrWeights, LayerWeights
from pathlib import Path

@pytest.fixture
def mock_weight_files(tmp_path):
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    (weight_dir / "tok_embeddings.weight.npy").touch()
    (weight_dir / "norm.weight.npy").touch()
    (weight_dir / "output.weight.npy").touch()
    for i in range(16):
        (weight_dir / f"layers.{i}.attention.wq.weight.npy").touch()
        (weight_dir / f"layers.{i}.attention.wk.weight.npy").touch()
        (weight_dir / f"layers.{i}.attention.wv.weight.npy").touch()
        (weight_dir / f"layers.{i}.attention.wo.weight.npy").touch()
        (weight_dir / f"layers.{i}.feed_forward.w1.weight.npy").touch()
        (weight_dir / f"layers.{i}.feed_forward.w2.weight.npy").touch()
        (weight_dir / f"layers.{i}.feed_forward.w3.weight.npy").touch()
        (weight_dir / f"layers.{i}.ffn_norm.weight.npy").touch()
        (weight_dir / f"layers.{i}.attention_norm.weight.npy").touch()
    return weight_dir

def test_load_weights(mock_weight_files, monkeypatch):
    def mock_load(file, mmap_mode, allow_pickle):
        return jnp.ones((10, 10))
    
    monkeypatch.setattr(jnp, "load", mock_load)
    
    weights = load_weights(ckpt_dir=mock_weight_files)
    assert isinstance(weights, XfmrWeights)
    assert len(weights.layer_weights) == 16
    assert isinstance(weights.layer_weights[0], LayerWeights)

def test_xfmr_weights_structure():
    weights = XfmrWeights(
        tok_embeddings=jnp.ones((100, 32)),
        norm=jnp.ones(32),
        output=jnp.ones((100, 32)),
        layer_weights=[LayerWeights(
            wq=jnp.ones((32, 32)),
            wk=jnp.ones((32, 32)),
            wv=jnp.ones((32, 32)),
            wo=jnp.ones((32, 32)),
            w1=jnp.ones((32, 32)),
            w2=jnp.ones((32, 32)),
            w3=jnp.ones((32, 32)),
            ffn_norm=jnp.ones(32),
            attention_norm=jnp.ones(32)
        )]
    )
    assert weights.tok_embeddings.shape == (100, 32)
    assert weights.norm.shape == (32,)
    assert weights.output.shape == (100, 32)
    assert len(weights.layer_weights) == 1
    assert weights.layer_weights[0].wq.shape == (32, 32)