from entropix.model import xfmr
from entropix.tokenizer import Tokenizer
from entropix.config import ModelParams, RopeParams
from entropix.weights import load_weights
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# tokenize the text to a token sequence
# run the token sequence through the model to get the attention entropy scores
# compute atten_entropy = atten_stats.entropy.sum(axis=-1).reshape(-1, atten_stats.entropy.shape[-2] * atten_stats.entropy.shape[-1]) # of shape (bsz, seq_len * n_layers)
# compute the short time fourier transform of atten_entropy
# display the text and above it the fourier transform such that each window is scaled to evenly fit within the width of the text, i.e. the number of characters in the text


def compute_stft(array, window_size):
    """
    Computes the Short-Time Fourier Transform (STFT) of the input array.

    Parameters:
    - array (jax.numpy.ndarray): The input array of float32 values.
    - window_size (int): The window size for the STFT.
    """
    f, t, Zxx = compute_stft(array, window_size)
    magnitude = np.abs(Zxx)
    return np.stft(array, window='hann', nperseg=window_size, noverlap=window_size//2)

def plot_stft_with_text(string, array, window_size):
    """
    Plots the Short-Time Fourier Transform (STFT) of the input array
    above the given string, aligning characters with the plot's x-axis.

    Parameters:
    - string (str): The string to display.
    - array (jax.numpy.ndarray): The input array of float32 values.
    - window_size (int): The window size for the STFT.
    """
    # Convert JAX array to NumPy
    array_np = np.array(array)



    # Determine alignment parameters
    N = len(string)        # Number of characters
    L = len(array_np)      # Number of samples
    K = L / N              # Samples per character

    # Generate x-axis for STFT plot aligned to number of characters
    time_axis = np.linspace(0, N, len(t))

    # Create the plot
    fig, (ax_stft, ax_text) = plt.subplots(2, 1, figsize=(12, 6),
                                          gridspec_kw={'height_ratios': [3, 1]},
                                          sharex=True)

    # Plot STFT magnitude
    im = ax_stft.pcolormesh(time_axis, f, magnitude, shading='gouraud')
    ax_stft.set_ylabel('Frequency [Hz]')
    ax_stft.set_title('Short-Time Fourier Transform')

    fig.colorbar(im, ax=ax_stft, format='%+2.0f dB')

    # Configure x-axis to represent characters
    ax_stft.set_xlim(0, N)
    ax_stft.set_xticks(np.arange(N) + 0.5)
    ax_stft.set_xticklabels([])  # Hide x-tick labels on STFT plot

    # Prepare text for the lower axis
    ax_text.axis('off')  # Hide the axis

    # Calculate character positions
    char_positions = np.linspace(0, N, len(string) + 1)
    for i, char in enumerate(string):
        # Position each character in the center of its segment
        ax_text.text((char_positions[i] + char_positions[i+1])/2, 0.5, char,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
  params = {
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
  }
  
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
  xfmr_weights = load_weights()
  #xfmr_weights = load_weights(ckpt_dir=Path('weights/1B-Base'))
  tokenizer = Tokenizer('entropix/tokenizer.model')
  text = "The quick brown fox jumps over the lazy dog"
  tokenized_text = tokenizer.tokenize(text)
    tokenized_text = xfmr(xfmr_weights, model_params, lm_state.context, lm_state.pos, freqs_cis=lm_state.freqs_cis[:seqlen], kvcache=lm_state.kvcache, attn_mask=lm_state.attn_mask)
    model = xfmr(tokenized_text)
    atten_stats = model.attention_stats
    atten_entropy = atten_stats.entropy.sum(axis=-1).reshape(-1, atten_stats.entropy.shape[-2] * atten_stats.entropy.shape[-1])
    plot_stft_with_text(text, atten_entropy, 10)
    