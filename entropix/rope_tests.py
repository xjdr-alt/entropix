from typing import Tuple
import jax.numpy as jnp
import jax
import math
from entropix.config import RopeParams
from functools import partial
from hypothesis import given, strategies as st
import numpy as np
import time

@partial(jax.jit,  static_argnames=("dtype"))
def rotate_every_two(x: jnp.ndarray, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    return jnp.stack([-x_odd, x_even], axis=-1).reshape(x.shape).astype(dtype)

@partial(jax.jit, static_argnames=("dtype"))
def apply_rotary_emb_alt(x: jnp.ndarray, freqs: jax.Array, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    sin = jnp.sin(freqs).astype(dtype)  
    cos = jnp.cos(freqs).astype(dtype) 
    sin = jnp.repeat(sin, 2, axis=-1)  
    cos = jnp.repeat(cos, 2, axis=-1)  
    result = (x * cos) + (rotate_every_two(x) * sin)
    return result

@partial(jax.jit, static_argnames=("dtype"))
def apply_rotary_emb(xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array, dtype: jnp.dtype = jnp.float32) -> Tuple[jax.Array, jax.Array]:
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    freqs_cis_reshaped = freqs_cis[None, :, :]
    xq_out = xq_ * freqs_cis_reshaped
    xk_out = xk_ * freqs_cis_reshaped
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1) 
    return xq_out.astype(dtype), xk_out.astype(dtype)

def precompute_freqs_cis(end: int, rope_params: RopeParams, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    freqs = 1.0 / (rope_params.theta ** (jnp.arange(0, rope_params.dim, 2)[: (rope_params.dim // 2)].astype(dtype) / rope_params.dim))
    if rope_params.use_scaled_rope:
        freqs = apply_scaling(rope_params, freqs)
    t = jnp.arange(end, dtype=dtype)
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1j * freqs)

def test_rotary_emb_equivalence(batch_size=4, max_seq_len=12, dim=4, dtype=jnp.float32):
    key = jax.random.PRNGKey(0)
    
    # Generate freqs as angles in radians
    angles = jax.random.uniform(key, (max_seq_len, dim // 2), minval=0.0, maxval=2.0 * jnp.pi, dtype=dtype)
    freqs_cis = jnp.exp(1j * angles)
    
    # Generate input tensors
    xq = jax.random.uniform(key, (batch_size, max_seq_len, dim), dtype=dtype)
    xk = jax.random.uniform(key, (batch_size, max_seq_len, dim), dtype=dtype)

    # Apply rotary embeddings using both implementations
    xq_alt = apply_rotary_emb_alt(xq, angles)
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
    
    # Assert equivalence for xq
    if jnp.allclose(xq_alt, xq_out, atol=1e-6):
        print("xq_alt and xq_out are equivalent.")
    else:
        max_diff = jnp.max(jnp.abs(xq_alt - xq_out))
        print(f"xq_alt and xq_out are not close: max difference = {max_diff}")

    print("Rotary embeddings equivalence test completed.")

def test_edge_cases():
    dtype = jnp.float32
    batch_size = 2
    max_seq_len = 4
    dim = 4
    key = jax.random.PRNGKey(42)

    # Define specific angles
    angles = jnp.array([
        [0.0, 0.0],                    # Zero rotation
        [2.0 * jnp.pi, 2.0 * jnp.pi],  # Full rotation
        [jnp.pi, jnp.pi],              # Half rotation
        [-jnp.pi / 2, -jnp.pi / 2],    # Negative rotation
    ], dtype=dtype)  # Shape: (4, 2)

    # Ensure angles match max_seq_len
    angles = angles[:max_seq_len]
    freqs_cis = jnp.exp(1j * angles)

    # Generate input tensors
    xq = jnp.array([
        [[1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [1.0, 1.0, 1.0, 1.0],
         [0.0, 0.0, 0.0, 0.0]]
    ] * batch_size)  # Shape: (2, 4, 4)

    xk = jnp.array([
        [[1.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 1.0],
         [1.0, -1.0, 1.0, -1.0],
         [0.5, 0.5, 0.5, 0.5]]
    ] * batch_size)  # Shape: (2, 4, 4)

    # Expected outputs for apply_rotary_emb_alt
    xq_alt_expected = jnp.array([
        [
            [1.0, 0.0, 1.0, 0.0],  # No rotation
            [-1.0, 0.0, -1.0, 0.0],  # 90 degrees rotation
            [1.0, 1.0, 1.0, 1.0],  # Rotation by pi radians
            [0.0, 0.0, 0.0, 0.0],  # No rotation
        ]
    ] * batch_size)

    # Apply rotary embeddings
    xq_alt = apply_rotary_emb_alt(xq, angles)
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

    # Trim or pad if necessary
    # Here, shapes already match

    # Assert equivalence for xq
    if jnp.allclose(xq_alt, xq_out, atol=1e-6):
        print("Edge Case Test: xq_alt and xq_out are equivalent.")
    else:
        max_diff = jnp.max(jnp.abs(xq_alt - xq_out))
        print(f"Edge Case Test: xq_alt and xq_out are not close: max difference = {max_diff}")

    print("Edge cases rotary embeddings test completed.")

@given(
    batch_size=st.integers(min_value=1, max_value=8),
    max_seq_len=st.integers(min_value=1, max_value=64),
    dim=st.sampled_from([2, 4, 8, 16]),
    angles=st.lists(
        st.lists(
            st.floats(min_value=0, max_value=2.0 * np.pi),
            min_size=2,
            max_size=2
        ),
        min_size=1,
        max_size=64
    )
)
def test_rotary_emb_hypothesis(batch_size, max_seq_len, dim, angles):
    dtype = jnp.float32
    key = jax.random.PRNGKey(0)

    # Adjust angles to match max_seq_len and dim//2
    angles = angles[:max_seq_len]
    angles = [angle[:dim//2] for angle in angles]
    angles = jnp.array(angles, dtype=dtype)

    # Compute complex exponentials
    freqs_cis = jnp.exp(1j * angles)

    # Generate input tensors with random values
    xq = jax.random.uniform(key, (batch_size, max_seq_len, dim), dtype=dtype)
    xk = jax.random.uniform(key, (batch_size, max_seq_len, dim), dtype=dtype)

    # Apply rotary embeddings
    xq_alt = apply_rotary_emb_alt(xq, angles)
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

    # Assert equivalence for xq
    assert jnp.allclose(xq_alt, xq_out, atol=1e-6), f"Mismatch with batch_size={batch_size}, max_seq_len={max_seq_len}, dim={dim}"
    # Similarly, you can assert for xk_out

def run_random_consistency_tests(num_tests=100):
    dtype = jnp.float32
    max_seq_len = 20
    batch_size = 8
    dim = 16
    key = jax.random.PRNGKey(123)

    for i in range(num_tests):
        # Split key for reproducibility
        subkey = jax.random.split(key, 2)
        key = subkey[0]

        # Generate random angles
        angles = jax.random.uniform(subkey[1], (max_seq_len, dim // 2), minval=0.0, maxval=2.0 * jnp.pi, dtype=dtype)
        freqs_cis = jnp.exp(1j * angles)

        # Generate random input tensors
        xq = jax.random.uniform(subkey[0], (batch_size, max_seq_len, dim), dtype=dtype)
        xk = jax.random.uniform(subkey[0], (batch_size, max_seq_len, dim), dtype=dtype)

        # Apply rotary embeddings
        xq_alt = apply_rotary_emb_alt(xq, angles)
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

        # Assert equivalence for xq
        if not jnp.allclose(xq_alt, xq_out, atol=1e-6):
            max_diff = jnp.max(jnp.abs(xq_alt - xq_out))
            print(f"Random Test {i+1}: xq_alt and xq_out are not close: max difference = {max_diff}")
            return
    print(f"All {num_tests} random consistency tests passed.")

def test_dimensionality_variation():
    dtype = jnp.float32
    batch_size = 2
    max_seq_len = 8
    dimensions = [2, 4, 6, 8, 10]  # Including both even and odd dimensions

    for dim in dimensions:
        print(f"\nTesting with dim={dim}")
        key = jax.random.PRNGKey(dim)  # Different seed for each dim

        # Adjust angles to match dim//2 (for even dimensions)
        # For odd dimensions, the last dimension might not be paired
        angles_dim = dim // 2 * 2  # Ensure even number
        angles = jax.random.uniform(key, (max_seq_len, angles_dim // 2), minval=0.0, maxval=2.0 * jnp.pi, dtype=dtype)
        freqs_cis = jnp.exp(1j * angles)

        # Generate input tensors with zero padding if dim is odd
        if dim % 2 != 0:
            # Pad the last dimension with zeros
            xq = jax.random.uniform(key, (batch_size, max_seq_len, dim), dtype=dtype)
            xk = jax.random.uniform(key, (batch_size, max_seq_len, dim), dtype=dtype)
            xq = jnp.pad(xq, ((0,0), (0,0), (0,1)), mode='constant')
            xk = jnp.pad(xk, ((0,0), (0,0), (0,1)), mode='constant')
        else:
            xq = jax.random.uniform(key, (batch_size, max_seq_len, dim), dtype=dtype)
            xk = jax.random.uniform(key, (batch_size, max_seq_len, dim), dtype=dtype)

        # Apply rotary embeddings
        xq_alt = apply_rotary_emb_alt(xq, angles)
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

        # Trim padding if necessary
        if dim % 2 != 0:
            xq_alt = xq_alt[..., :-1]
            xq_out = xq_out[..., :-1]

        # Assert equivalence
        if jnp.allclose(xq_alt, xq_out, atol=1e-6):
            print(f"Dimensionality Test {dim}: xq_alt and xq_out are equivalent.")
        else:
            max_diff = jnp.max(jnp.abs(xq_alt - xq_out))
            print(f"Dimensionality Test {dim}: xq_alt and xq_out are not close: max difference = {max_diff}")

    print("Dimensionality variation rotary embeddings tests completed.")

def test_known_inputs():
    dtype = jnp.float32
    batch_size = 1
    max_seq_len = 2
    dim = 4
    key = jax.random.PRNGKey(1)

    # Define specific angles
    angles = jnp.array([
        [0.0, jnp.pi / 2],  # 0 radians and 90 degrees
        [jnp.pi, 3 * jnp.pi / 2],  # 180 degrees and 270 degrees
    ], dtype=dtype)  # Shape: (2, 2)

    # Compute complex exponentials
    freqs_cis = jnp.exp(1j * angles)  # Shape: (2, 2)

    # Define input tensors
    xq = jnp.array([
        [[1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0]]
    ])  # Shape: (1, 2, 4)

    xk = jnp.array([
        [[1.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 1.0]]
    ])  # Shape: (1, 2, 4)

    # Expected outputs for apply_rotary_emb_alt
    xq_alt_expected = jnp.array([
        [
            [1.0, 0.0, 1.0, 0.0],  # No rotation
            [-1.0, 0.0, -1.0, 0.0],  # 90 degrees rotation
        ]
    ])  # Shape: (1, 2, 4)

    # Apply rotary embeddings
    xq_alt = apply_rotary_emb_alt(xq, angles)
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

    # Assert equivalence for xq
    if jnp.allclose(xq_alt, xq_alt_expected, atol=1e-6):
        print("Known Input Test: xq_alt matches expected.")
    else:
        print("Known Input Test: xq_alt does not match expected.")

    if jnp.allclose(xq_out, xq_alt_expected, atol=1e-6):
        print("Known Input Test: xq_out matches expected.")
    else:
        print("Known Input Test: xq_out does not match expected.")

    print("Known inputs rotary embeddings test completed.")

def run_random_consistency_tests(num_tests=100):
    dtype = jnp.float32
    max_seq_len = 20
    batch_size = 8
    dim = 16
    key = jax.random.PRNGKey(123)

    for i in range(num_tests):
        # Split key for reproducibility
        subkey = jax.random.split(key, 2)
        key = subkey[0]

        # Generate random angles
        angles = jax.random.uniform(subkey[1], (max_seq_len, dim // 2), minval=0.0, maxval=2.0 * jnp.pi, dtype=dtype)
        freqs_cis = jnp.exp(1j * angles)

        # Generate random input tensors
        xq = jax.random.uniform(subkey[0], (batch_size, max_seq_len, dim), dtype=dtype)
        xk = jax.random.uniform(subkey[0], (batch_size, max_seq_len, dim), dtype=dtype)

        # Apply rotary embeddings
        xq_alt = apply_rotary_emb_alt(xq, angles)
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

        # Assert equivalence for xq
        if not jnp.allclose(xq_alt, xq_out, atol=1e-6):
            max_diff = jnp.max(jnp.abs(xq_alt - xq_out))
            print(f"Random Test {i+1}: xq_alt and xq_out are not close: max difference = {max_diff}")
            return
    print(f"All {num_tests} random consistency tests passed.")

def benchmark_rotary_embeddings():
    dtype = jnp.float32
    batch_size = 128
    max_seq_len = 512
    dim = 64
    key = jax.random.PRNGKey(0)

    # Generate angles and complex exponentials
    angles = jax.random.uniform(key, (max_seq_len, dim // 2), minval=0.0, maxval=2.0 * jnp.pi, dtype=dtype)
    freqs_cis = jnp.exp(1j * angles)

    # Generate input tensors
    xq = jax.random.uniform(key, (batch_size, max_seq_len, dim), dtype=dtype)
    xk = jax.random.uniform(key, (batch_size, max_seq_len, dim), dtype=dtype)

    # Warm-up JIT
    apply_rotary_emb(xq, xk, freqs_cis)

    # Time apply_rotary_emb_alt
    start_time = time.time()
    xq_alt = apply_rotary_emb_alt(xq, angles)
    elapsed_alt = time.time() - start_time
    print(f"apply_rotary_emb_alt took {elapsed_alt:.6f} seconds.")

    # Time apply_rotary_emb
    start_time = time.time()
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
    elapsed_emb = time.time() - start_time
    print(f"apply_rotary_emb took {elapsed_emb:.6f} seconds.")

def main():
    print("Running equivalence tests with predefined parameters...")
    test_rotary_emb_equivalence()

    print("\nRunning edge cases tests...")
    test_edge_cases()

    print("\nRunning dimensionality variation tests...")
    test_dimensionality_variation()

    print("\nRunning known input tests...")
    test_known_inputs()

    print("\nRunning randomized consistency tests...")
    run_random_consistency_tests(num_tests=100)

    print("\nRunning performance benchmarks...")
    benchmark_rotary_embeddings()

if __name__ == "__main__":
    main()