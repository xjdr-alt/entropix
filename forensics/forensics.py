import jax.numpy as jnp
import numpy as np
from entropix.model import xfmr
from entropix.config import ModelParams, RopeParams
from entropix.rope import precompute_freqs_cis
from entropix.kvcache import KVCache
from entropix.generator import build_attn_mask
from entropix.weights import load_weights
from entropix.tokenizer import Tokenizer
from entropix.lm_state import LMState
import argparse
from rich.console import Console
from rich.text import Text
import colorsys

def compute_stft_custom(entropy_array, window_size=64, step_size=16):
    """
    Computes the Short-Time Fourier Transform (STFT) using a Hamming window.

    Parameters:
    - entropy_array (numpy.ndarray): 1D array of aggregated attention entropy values per token.
    - window_size (int): Size of each window for FFT.
    - step_size (int): Step size between consecutive windows.

    Returns:
    - stft_matrix (numpy.ndarray): 2D array of FFT magnitudes (frequency bins x time windows).
    - x_values_stft (numpy.ndarray): 1D array representing the center index of each window.
    """
    window_fn = np.hamming(window_size)
    stft_magnitudes = []
    x_values_stft = []

    for i in range(0, len(entropy_array) - window_size + 1, step_size):
        window = entropy_array[i:i + window_size]
        windowed = window * window_fn
        fft_result = np.fft.fft(windowed)
        magnitudes = np.abs(fft_result)[:window_size // 2]  # Take positive frequencies
        stft_magnitudes.append(magnitudes)
        # Center index of the window
        x_center = i + window_size // 2
        x_values_stft.append(x_center)

    stft_matrix = np.array(stft_magnitudes).T  # Shape: (frequency_bins, time_windows)
    x_values_stft = np.array(x_values_stft)
    return stft_matrix, x_values_stft

def compute_spectral_centroid(stft_matrix):
    """
    Computes the spectral centroid for each time window in the STFT matrix.

    Parameters:
    - stft_matrix (numpy.ndarray): 2D array of FFT magnitudes (frequency bins x time windows).

    Returns:
    - spectral_centroids (numpy.ndarray): 1D array of spectral centroid values per window.
    """
    frequency_bins, time_windows = stft_matrix.shape
    # Normalize frequency indices to [0, 1]
    f = np.linspace(0, 1, frequency_bins)
    # Avoid division by zero
    magnitude_sums = stft_matrix.sum(axis=0)
    spectral_centroids = np.zeros(time_windows)
    nonzero = magnitude_sums > 0
    spectral_centroids[nonzero] = np.sum(f[:, np.newaxis] * stft_matrix[:, nonzero], axis=0) / magnitude_sums[nonzero]
    return spectral_centroids

def map_centroid_to_color(centroids):
    """
    Maps normalized spectral centroid values to hexadecimal RGB colors.

    Parameters:
    - centroids (numpy.ndarray): 1D array of spectral centroid values normalized between 0 and 1.

    Returns:
    - colors (list of str): List of hexadecimal color codes.
    """
    colors = []
    for c in centroids:
        # Ensure centroid is within [0, 1]
        c = np.clip(c, 0, 1)
        # Map centroid to hue (0-360 degrees)
        hue = c * 360  # 0 to 360
        saturation = 1.0  # Full saturation
        value = 1.0  # Full brightness
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue / 360, saturation, value)
        # Convert RGB to hexadecimal
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

def interpolate_centroids(spectral_centroids, num_chars):
    """
    Interpolates spectral centroid values to match the number of characters.

    Parameters:
    - spectral_centroids (numpy.ndarray): 1D array of spectral centroid values per window.
    - num_chars (int): Total number of characters in the text.

    Returns:
    - interpolated_centroids (numpy.ndarray): 1D array of spectral centroid values per character.
    """
    time_windows = len(spectral_centroids)
    if time_windows == 0:
        return np.zeros(num_chars)
    
    # Positions of spectral centroids in character indices
    centroid_positions = np.linspace(0, num_chars - 1, time_windows)
    
    # Character positions
    char_positions = np.arange(num_chars)
    
    # Interpolate
    interpolated_centroids = np.interp(char_positions, centroid_positions, spectral_centroids)
    
    return interpolated_centroids

def assign_colors_to_tokens(interpolated_centroids):
    """
    Assigns colors to each character based on interpolated spectral centroid values.

    Parameters:
    - interpolated_centroids (numpy.ndarray): 1D array of spectral centroid values per character.

    Returns:
    - token_colors (list of str): List of hexadecimal color codes corresponding to each character.
    """
    # Normalize the interpolated centroids to [0, 1]
    min_val = interpolated_centroids.min()
    max_val = interpolated_centroids.max()
    if max_val - min_val == 0:
        normalized_centroids = np.zeros_like(interpolated_centroids)
    else:
        normalized_centroids = (interpolated_centroids - min_val) / (max_val - min_val)
    
    # Debugging: Print min and max after normalization
    print(f"Normalized Centroids Min: {normalized_centroids.min()}, Max: {normalized_centroids.max()}, Mean: {normalized_centroids.mean()}, Std: {normalized_centroids.std()}")
    
    # Map normalized values to colors
    colors = map_centroid_to_color(normalized_centroids)
    
    return colors

def create_colored_text(text, token_colors):
    """
    Creates colored text using the Rich library based on token_colors.

    Parameters:
    - text (str): The input text.
    - token_colors (list of str): List of hexadecimal color codes corresponding to each character.
    """
    console = Console()
    colored_text = Text()

    for idx, char in enumerate(text):
        color = token_colors[idx] if idx < len(token_colors) else '#FFFFFF'  # Default to white
        # Apply the color as a style
        colored_text.append(char, style=f"color({color})")

    console.print(colored_text)

def analyze_atten_entropy(xfmr_weights, model_params, tokenizer, text):
    """
    Analyzes attention entropy and visualizes it by coloring text characters based on STFT spectral centroid values.

    Parameters:
    - xfmr_weights: Transformer model weights.
    - model_params: Model parameters.
    - tokenizer: Tokenizer instance.
    - text (str): The input text to analyze.
    """
    tokens = tokenizer.encode(text, bos=False, eos=False, allowed_special='all')
    tokens = np.array([tokens])
    seqlen = tokens.shape[-1]
    n_words = tokenizer.n_words
    lm_state = LMState(
        prompt=tokens,
        logits=jnp.zeros((1, n_words), dtype=jnp.bfloat16),
        freqs_cis=precompute_freqs_cis(
            head_dim=model_params.head_dim,
            max_seq_len=model_params.max_seq_len,
            rope_params=model_params.rope_params
        ),
        kvcache=KVCache.new(
            model_params.n_layers,
            1,
            model_params.max_seq_len,
            model_params.n_local_kv_heads,
            model_params.head_dim
        ),
        attn_mask=build_attn_mask(seqlen, 0),
        gen_tokens=jnp.zeros((1, 0), dtype=jnp.int32),
        state=jnp.zeros((1, 1), dtype=jnp.int32),
        pos=0
    )
    _, _, attn_stats = xfmr(
        xfmr_weights,
        model_params,
        lm_state.prompt,
        lm_state.pos,
        freqs_cis=lm_state.freqs_cis[:seqlen],
        kvcache=lm_state.kvcache,
        attn_mask=lm_state.attn_mask
    )

    # Debugging: Check the shape of attn_stats.entropy
    print(f"Original Attention Entropy Shape: {attn_stats.entropy.shape}")

    # Aggregate attention entropy into a 1D array per token by summing across heads and layers
    if attn_stats.entropy.ndim == 4:
        # Sum over heads and layers to get per-token entropy
        attn_entropy = attn_stats.entropy.sum(axis=(2, 3)).reshape(-1)  # Shape: (batch * seq_len,)
    elif attn_stats.entropy.ndim == 3:
        # Sum over heads to get per-token entropy
        attn_entropy = attn_stats.entropy.sum(axis=2).reshape(-1)  # Shape: (batch * seq_len,)
    else:
        raise ValueError(f"Unexpected entropy array shape: {attn_stats.entropy.shape}")

    # Convert from JAX array to NumPy
    attn_entropy = np.array(attn_entropy)

    # Debugging: Check the length and statistics of attn_entropy
    print(f"Aggregated Attention Entropy Shape: {attn_entropy.shape}")
    print(f"Attention Entropy Min: {attn_entropy.min()}, Max: {attn_entropy.max()}, Mean: {attn_entropy.mean()}, Std: {attn_entropy.std()}")

    # Check variance to ensure meaningful spectral analysis
    if attn_entropy.std() < 1e-6:
        print("Attention entropy has very low variance. Applying scaling to enhance variation.")
        # Apply logarithmic scaling to increase variance
        attn_entropy = np.log(attn_entropy + 1e-6)
        print(f"After Log Scaling - Min: {attn_entropy.min()}, Max: {attn_entropy.max()}, Mean: {attn_entropy.mean()}, Std: {attn_entropy.std()}")

    # Verify variance after scaling
    print(f"Post-Scaling Attention Entropy Min: {attn_entropy.min()}, Max: {attn_entropy.max()}, Mean: {attn_entropy.mean()}, Std: {attn_entropy.std()}")

    # Compute STFT using adjusted window_size and step_size for finer resolution
    window_size = 64   # Smaller window size for finer spectral analysis
    step_size = 16     # Smaller step size for increased overlap and smoother transitions
    stft_matrix, x_values_stft = compute_stft_custom(attn_entropy, window_size, step_size)

    # Debugging: Check the shape of stft_matrix and x_values_stft
    print(f"STFT Matrix Shape: {stft_matrix.shape}")       # (frequency_bins, time_windows)
    print(f"X Values STFT Shape: {x_values_stft.shape}") # (time_windows,)

    if stft_matrix.size == 0:
        print("No STFT magnitudes computed. Check window_size and step_size.")
        return

    # Compute spectral centroid for each window
    spectral_centroids = compute_spectral_centroid(stft_matrix)

    # Debugging: Check spectral centroids
    print(f"Spectral Centroids: {spectral_centroids}")
    print(f"Spectral Centroids Min: {spectral_centroids.min()}, Max: {spectral_centroids.max()}, Mean: {spectral_centroids.mean()}, Std: {spectral_centroids.std()}")

    # Interpolate spectral centroids to match the number of characters
    num_chars = len(text)
    interpolated_centroids = interpolate_centroids(spectral_centroids, num_chars)

    # Debugging: Check interpolated centroids
    print(f"Interpolated Spectral Centroids Shape: {interpolated_centroids.shape}")
    print(f"Interpolated Centroids Sample: {interpolated_centroids[:10]}")  # Show first 10 for brevity

    # Assign colors to tokens based on interpolated spectral centroids
    token_colors = assign_colors_to_tokens(interpolated_centroids)

    # Debugging: Check a sample of assigned colors
    print(f"Assigned Colors Sample: {token_colors[:10]}")  # Show first 10 for brevity

    # Create and display the colored text
    create_colored_text(text, token_colors)

def load_model(tokenizer_path, params={
    "dim": 2048,
    "n_layers": 16,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_words": 128256,
    "ffn_dim_multiplier": 1.5,
    "multiple_of": 256,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "scale_factor": 8,
    "low_freq_factor": 1,
    "high_freq_factor": 4,
    "old_context_len": 8192,
    "use_scaled_rope": True,
    "max_seq_len": 4096
}):
    """
    Loads the transformer model, tokenizer, and weights.

    Parameters:
    - tokenizer_path (str): Path to the tokenizer model.
    - params (dict): Model parameters.

    Returns:
    - xfmr_weights: Transformer model weights.
    - model_params: Model parameters.
    - tokenizer: Tokenizer instance.
    """
    LLAMA_1B_ROPE = RopeParams(
        rope_theta=params["rope_theta"],
        use_scaled_rope=params["use_scaled_rope"],
        scale_factor=params["scale_factor"],
        low_freq_factor=params["low_freq_factor"],
        high_freq_factor=params["high_freq_factor"],
        old_context_len=params["old_context_len"]
    )
    LLAMA_1B_PARAMS = ModelParams(
        n_layers=params["n_layers"],
        n_local_heads=params["n_heads"],
        n_local_kv_heads=params["n_kv_heads"],
        head_dim=params["dim"] // params["n_heads"],
        max_seq_len=params["max_seq_len"],
        rope_params=LLAMA_1B_ROPE,
        d_model=params["dim"]
    )
    model_params = LLAMA_1B_PARAMS
    tokenizer = Tokenizer(model_path=tokenizer_path)
    xfmr_weights = load_weights()
    return xfmr_weights, model_params, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize attention entropy by coloring text characters based on STFT spectral centroid values.")
    parser.add_argument("txt_file", help="Path to the .txt file to analyze")
    args = parser.parse_args()

    # Load the model
    tokenizer_path = "./entropix/tokenizer.model"  # Update this path if necessary
    try:
        xfmr_weights, model_params, tokenizer = load_model(tokenizer_path)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # Read the text file
    try:
        with open(args.txt_file, 'r') as file:
            text = file.read().strip()
    except Exception as e:
        print(f"Error reading the text file: {e}")
        return

    if not text:
        print("The input text file is empty. Please provide valid text.")
        return

    # Analyze attention entropy and visualize
    analyze_atten_entropy(xfmr_weights, model_params, tokenizer, text)

if __name__ == "__main__":
    main()
