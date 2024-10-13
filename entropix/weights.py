from typing import List, NamedTuple

from typing import List, NamedTuple

import jax
import jax.numpy as jnp

from pathlib import Path

# Assume ModelParams is defined elsewhere and provides necessary parameters
# from entropix.config import ModelParams

class LayerWeights(NamedTuple):
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    w1: jax.Array
    w2: jax.Array
    w3: jax.Array
    ffn_norm: jax.Array
    attention_norm: jax.Array

class XfmrWeights(NamedTuple):
    tok_embeddings: jax.Array
    norm: jax.Array
    output: jax.Array
    layer_weights: List[LayerWeights]

def initialize_packed_weights(shape, key):
    """
    Initialize weights using sphere packing.

    Pros:
    - Ensures weights are maximally distant in the vector space, reducing correlation.
    - Can improve the model's ability to distinguish between different inputs.

    Cons:
    - Requires retraining the model, as pre-trained weights are not used.
    - May lead to unstable training if not carefully managed.
    """
    num_vectors, dimensions = shape

    # Generate random points on a unit sphere
    random_points = jax.random.normal(key, (num_vectors, dimensions))
    random_points /= jnp.linalg.norm(random_points, axis=1, keepdims=True) + 1e-8  # Avoid division by zero

    # Optionally apply scaling to spread out the weights further (commented out here)
    # scale_factors = jnp.linspace(0.8, 1.2, num_vectors).reshape(num_vectors, 1)
    # packed_weights = random_points * scale_factors

    # For simplicity, we can omit scaling and use the normalized points directly
    packed_weights = random_points
    return packed_weights

def load_weights(ckpt_dir: Path, model_params, use_sphere_packing: bool = True):
    """
    Load transformer weights, with an option to initialize using sphere packing.

    Pros of using sphere packing:
    - Promotes uncorrelated weight initialization.
    - May enhance the model's handling of uncertainty.

    Cons:
    - Requires retraining the model from scratch.
    - Discards any pre-trained knowledge from existing weights.
    """
    w = {}
    layer_weights = []

    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        print("GPU not found. Using CPU instead.")
        device = jax.devices("cpu")[0]

    if use_sphere_packing:
        print("Initializing weights using sphere packing...")
        key = jax.random.PRNGKey(0)

        # Initialize token embeddings
        tok_embeddings_shape = (model_params.vocab_size, model_params.dim)
        key, subkey = jax.random.split(key)
        w['tok_embeddings.weight'] = initialize_packed_weights(tok_embeddings_shape, subkey)

        # Initialize weights for each layer
        for i in range(model_params.n_layers):
            # Split the random key for reproducibility
            key, subkey = jax.random.split(key)
            layer_key = subkey

            # Define shapes for weight matrices
            layer_wq_shape = (model_params.dim, model_params.n_heads * model_params.head_dim)
            layer_wk_shape = (model_params.dim, model_params.n_heads * model_params.head_dim)
            layer_wv_shape = (model_params.dim, model_params.n_heads * model_params.head_dim)
            layer_wo_shape = (model_params.n_heads * model_params.head_dim, model_params.dim)

            layer_w1_shape = (model_params.dim, model_params.ffn_dim)
            layer_w2_shape = (model_params.ffn_dim, model_params.dim)
            layer_w3_shape = (model_params.dim, model_params.ffn_dim)

            # Initialize attention weights using sphere packing
            w[f'layers.{i}.attention.wq.weight'] = initialize_packed_weights(layer_wq_shape, layer_key)
            key, layer_key = jax.random.split(key)
            w[f'layers.{i}.attention.wk.weight'] = initialize_packed_weights(layer_wk_shape, layer_key)
            key, layer_key = jax.random.split(key)
            w[f'layers.{i}.attention.wv.weight'] = initialize_packed_weights(layer_wv_shape, layer_key)
            key, layer_key = jax.random.split(key)
            w[f'layers.{i}.attention.wo.weight'] = initialize_packed_weights(layer_wo_shape, layer_key)

            # Initialize feed-forward weights using sphere packing
            key, layer_key = jax.random.split(key)
            w[f'layers.{i}.feed_forward.w1.weight'] = initialize_packed_weights(layer_w1_shape, layer_key)
            key, layer_key = jax.random.split(key)
            w[f'layers.{i}.feed_forward.w2.weight'] = initialize_packed_weights(layer_w2_shape, layer_key)
            key, layer_key = jax.random.split(key)
            w[f'layers.{i}.feed_forward.w3.weight'] = initialize_packed_weights(layer_w3_shape, layer_key)

            # Initialize layer normalization weights to ones
            w[f'layers.{i}.ffn_norm.weight'] = jnp.ones((model_params.dim,))
            w[f'layers.{i}.attention_norm.weight'] = jnp.ones((model_params.dim,))
    else:
        print("Loading pre-trained weights from checkpoint...")
        for file in ckpt_dir.glob("*.npy"):
            name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
            weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
            w[name] = jax.device_put(weight, device)

    # Build the list of layer weights
    for i in range(model_params.n_layers):
        layer_weights.append(LayerWeights(
            wq=w[f'layers.{i}.attention.wq.weight'],
            wk=w[f'layers.{i}.attention.wk.weight'],
            wv=w[f'layers.{i}.attention.wv.weight'],
            wo=w[f'layers.{i}.attention.wo.weight'],
            w1=w[f'layers.{i}.feed_forward.w1.weight'],
            w2=w[f'layers.{i}.feed_forward.w2.weight'],
            w3=w[f'layers.{i}.feed_forward.w3.weight'],
            ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
            attention_norm=w[f'layers.{i}.attention_norm.weight'],
        ))

    if use_sphere_packing:
        # Initialize final layer norm and output weights
        w['norm.weight'] = jnp.ones((model_params.dim,))
        key, subkey = jax.random.split(key)
        output_shape = (model_params.vocab_size, model_params.dim)
        w['output.weight'] = initialize_packed_weights(output_shape, subkey)
    else:
        # Use pre-trained final layer norm and output weights
        w['norm.weight'] = w['norm.weight']
        w['output.weight'] = w['output.weight']

    # Create the transformer weights object
    xfmr_weights = XfmrWeights(
        tok_embeddings=w['tok_embeddings.weight'],
        norm=w['norm.weight'],
        output=w['output.weight'],
        layer_weights=layer_weights
    )

    return xfmr_weights
