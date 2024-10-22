import jax
import jax.numpy as jnp
import time
from functools import partial
import numpy as np
import pandas as pd
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from typing import Tuple, List, Dict, NamedTuple
import json
from datetime import datetime
import os

class LayerParams(NamedTuple):
    # Attention parameters
    q_weight: jnp.ndarray
    k_weight: jnp.ndarray
    v_weight: jnp.ndarray
    o_weight: jnp.ndarray
    # MLP parameters
    mlp1_weight: jnp.ndarray
    mlp2_weight: jnp.ndarray
    # Layer norms
    ln1_scale: jnp.ndarray
    ln1_bias: jnp.ndarray
    ln2_scale: jnp.ndarray
    ln2_bias: jnp.ndarray

def create_layer_params(rng: jax.random.PRNGKey, hidden_dim: int, mlp_dim: int) -> LayerParams:
    """Create parameters for a single layer."""
    keys = jax.random.split(rng, 6)
    return LayerParams(
        q_weight=jax.random.normal(keys[0], (hidden_dim, hidden_dim)) / np.sqrt(hidden_dim),
        k_weight=jax.random.normal(keys[1], (hidden_dim, hidden_dim)) / np.sqrt(hidden_dim),
        v_weight=jax.random.normal(keys[2], (hidden_dim, hidden_dim)) / np.sqrt(hidden_dim),
        o_weight=jax.random.normal(keys[3], (hidden_dim, hidden_dim)) / np.sqrt(hidden_dim),
        mlp1_weight=jax.random.normal(keys[4], (hidden_dim, mlp_dim)) / np.sqrt(hidden_dim),
        mlp2_weight=jax.random.normal(keys[5], (mlp_dim, hidden_dim)) / np.sqrt(mlp_dim),
        ln1_scale=jnp.ones(hidden_dim),
        ln1_bias=jnp.zeros(hidden_dim),
        ln2_scale=jnp.ones(hidden_dim),
        ln2_bias=jnp.zeros(hidden_dim)
    )

def stack_params(params_list: List[LayerParams]) -> LayerParams:
    """Stack parameters from multiple layers into arrays."""
    return LayerParams(
        q_weight=jnp.stack([p.q_weight for p in params_list]),
        k_weight=jnp.stack([p.k_weight for p in params_list]),
        v_weight=jnp.stack([p.v_weight for p in params_list]),
        o_weight=jnp.stack([p.o_weight for p in params_list]),
        mlp1_weight=jnp.stack([p.mlp1_weight for p in params_list]),
        mlp2_weight=jnp.stack([p.mlp2_weight for p in params_list]),
        ln1_scale=jnp.stack([p.ln1_scale for p in params_list]),
        ln1_bias=jnp.stack([p.ln1_bias for p in params_list]),
        ln2_scale=jnp.stack([p.ln2_scale for p in params_list]),
        ln2_bias=jnp.stack([p.ln2_bias for p in params_list])
    )

def create_model_params(rng: jax.random.PRNGKey, hidden_dim: int, num_layers: int, 
                       num_heads: int, mlp_factor: int = 4) -> LayerParams:
    """Create parameters for all layers."""
    mlp_dim = hidden_dim * mlp_factor
    keys = jax.random.split(rng, num_layers)
    layers = [create_layer_params(key, hidden_dim, mlp_dim) for key in keys]
    
    params_per_layer = (
        4 * hidden_dim * hidden_dim +  # Attention
        2 * hidden_dim * mlp_dim +     # MLP
        4 * hidden_dim                 # Layer norms
    )
    total_params = params_per_layer * num_layers
    
    print(f"\nModel configuration:")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Number of layers: {num_layers}")
    print(f"Number of heads: {num_heads}")
    print(f"MLP factor: {mlp_factor}")
    print(f"Total parameters: {total_params:,}")
    
    return stack_params(layers)

@jax.jit
def layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + 1e-5) + bias

@partial(jax.jit, static_argnames=('num_heads',))
def transformer_layer(x: jnp.ndarray, layer_idx: int, params: LayerParams, num_heads: int) -> jnp.ndarray:
    """Single transformer layer forward pass."""
    # Layer norm 1
    h = layer_norm(x, params.ln1_scale[layer_idx], params.ln1_bias[layer_idx])
    
    # Self-attention
    batch_size, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    
    # Compute QKV
    q = (h @ params.q_weight[layer_idx]).reshape(batch_size, seq_len, num_heads, head_dim)
    k = (h @ params.k_weight[layer_idx]).reshape(batch_size, seq_len, num_heads, head_dim)
    v = (h @ params.v_weight[layer_idx]).reshape(batch_size, seq_len, num_heads, head_dim)
    
    # Attention
    scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) / np.sqrt(head_dim)
    attn = jax.nn.softmax(scores, axis=-1)
    attended = jnp.einsum('bhqk,bkhd->bqhd', attn, v)
    
    attended = attended.reshape(batch_size, seq_len, hidden_dim)
    attended = attended @ params.o_weight[layer_idx]
    
    # Residual
    x = x + attended
    
    # Layer norm 2
    h = layer_norm(x, params.ln2_scale[layer_idx], params.ln2_bias[layer_idx])
    
    # MLP
    h = h @ params.mlp1_weight[layer_idx]
    h = jax.nn.gelu(h)
    h = h @ params.mlp2_weight[layer_idx]
    
    # Residual
    return x + h

@partial(jax.jit, static_argnames=('num_layers', 'num_heads'))
def model_forward(x: jnp.ndarray, params: LayerParams, num_layers: int, num_heads: int) -> jnp.ndarray:
    """Full model forward pass."""
    for i in range(num_layers):
        x = transformer_layer(x, i, params, num_heads)
    return x

def benchmark_mesh_config(
    mesh_shape: Tuple[int, ...],
    batch_size: int = 1,
    seq_length: int = 512,
    hidden_dim: int = 2048,
    num_layers: int = 24,
    num_heads: int = 32,
    num_warmup: int = 5,
    num_runs: int = 20
) -> dict:
    """Benchmark a specific mesh configuration."""
    print(f"\n{'='*60}")
    print(f"Testing mesh configuration: {mesh_shape}")
    print(f"{'='*60}")
    
    # Create model parameters
    rng = jax.random.PRNGKey(0)
    params = create_model_params(rng, hidden_dim, num_layers, num_heads)
    
    # Create mesh
    devices = mesh_utils.create_device_mesh(mesh_shape)
    with Mesh(devices, ('mp', 'fsdp')):
        # Create input data
        x = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_length, hidden_dim))
        
        # Warmup runs
        print("Performing warmup runs...")
        for i in range(num_warmup):
            _ = model_forward(x, params, num_layers, num_heads)
            print(f"Warmup {i+1}/{num_warmup}", end='\r')
        print("\nWarmup complete.")
        
        # Timed runs
        print(f"\nRunning {num_runs} benchmarks...")
        latencies = []
        jax.block_until_ready(x)
        
        for i in range(num_runs):
            start_time = time.time()
            output = model_forward(x, params, num_layers, num_heads)
            jax.block_until_ready(output)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            print(f"Run {i+1}/{num_runs}: {latency:.2f}ms", end='\r')
        
        print("\nBenchmark complete.")
            
    results = {
        'mesh_shape': f"{mesh_shape}",
        'mean_latency': float(np.mean(latencies)),
        'std_latency': float(np.std(latencies)),
        'min_latency': float(np.min(latencies)),
        'max_latency': float(np.max(latencies)),
        'p90_latency': float(np.percentile(latencies, 90)),
        'p99_latency': float(np.percentile(latencies, 99)),
        'raw_latencies': latencies,
        'config': {
            'batch_size': batch_size,
            'seq_length': seq_length,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads
        }
    }
    
    print("\nResults:")
    print(f"Mean latency: {results['mean_latency']:.2f}ms")
    print(f"Std dev: {results['std_latency']:.2f}ms")
    print(f"P90 latency: {results['p90_latency']:.2f}ms")
    print(f"P99 latency: {results['p99_latency']:.2f}ms")
    
    return results

def run_all_benchmarks(output_dir="benchmark_results"):
    """Run benchmarks for different mesh configurations and save results."""
    mesh_configs = [
        (4, 1),  # Pure model parallel
        (2, 2),  # Balanced
        (1, 4),  # Pure FSDP
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = []
    for mesh_shape in mesh_configs:
        result = benchmark_mesh_config(mesh_shape)
        results.append(result)
    
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['raw_latencies', 'config']} 
                             for r in results])
    
    csv_path = os.path.join(output_dir, f"mesh_benchmark_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    
    json_path = os.path.join(output_dir, f"mesh_benchmark_detailed_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Final Comparison:")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to:")
    print(f"CSV: {csv_path}")
    print(f"Detailed JSON: {json_path}")
    
    return results_df

if __name__ == "__main__":
    run_all_benchmarks()