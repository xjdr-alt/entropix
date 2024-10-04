import time
import psutil
import jax
import jax.numpy as jnp

def measure_inference_speed(model, input_data, num_runs=10):
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.generate(input_data)
    end_time = time.time()
    return (end_time - start_time) / num_runs

def measure_memory_usage(model, input_data):
    before_mem = psutil.virtual_memory().used
    _ = model.generate(input_data)
    after_mem = psutil.virtual_memory().used
    return after_mem - before_mem

def measure_scaling_efficiency(model, input_data, batch_sizes):
    results = {}
    for batch_size in batch_sizes:
        batched_input = jax.tree_map(lambda x: jnp.repeat(x, batch_size, axis=0), input_data)
        speed = measure_inference_speed(model, batched_input)
        results[batch_size] = speed
    return results

def run_benchmarks(model, input_data):
    return {
        "inference_speed": measure_inference_speed(model, input_data),
        "memory_usage": measure_memory_usage(model, input_data),
        "scaling_efficiency": measure_scaling_efficiency(model, input_data, [1, 2, 4, 8])
    }