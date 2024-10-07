import os
import subprocess
from tkinter import scrolledtext, ttk
import tkinter as tk
import threading
import jax
import jax.numpy as jnp
from entropix.config import LLAMA_1B_PARAMS
from entropix.weights import load_weights
from entropix.tokenizer import Tokenizer
from entropix.kvcache import KVCache
from entropix.main import sample, precompute_freqs_cis, build_attn_mask
from entropix.model import xfmr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from entropix.generator import generate

import numpy as np

class TextCompletionUI:
    def __init__(self, master):
        self.master = master
        master.title("Alien Communication Device")

        # Log GPU memory at initialization
        self.log_gpu_memory()

        # Initialize model components
        self.model_params = LLAMA_1B_PARAMS
        self.xfmr_weights = load_weights()
        self.tokenizer = Tokenizer('./entropix/tokenizer.model')
        
        # Create UI elements
        self.create_widgets()

        # Initialize generation state
        self.is_generating = False
        self.stop_generation = False
        self.generation_thread = None

        # Initialize list to store entropy values
        self.entropy_values = []

        # Bind the window close event to ensure graceful shutdown
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def log_gpu_memory(self):
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))

    def create_widgets(self):
        # Input text area
        self.input_label = ttk.Label(self.master, text="Enter your prompt:")
        self.input_label.pack(pady=5)
        self.input_text = scrolledtext.ScrolledText(self.master, height=5)
        self.input_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Generate button
        self.generate_button = ttk.Button(self.master, text="Generate", command=self.start_generation)
        self.generate_button.pack(pady=10)

        # Stop button (initially disabled)
        self.stop_button = ttk.Button(self.master, text="Stop", command=self.stop_generation_flag, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        # Output text area
        self.output_label = ttk.Label(self.master, text="Generated text:")
        self.output_label.pack(pady=5)
        self.output_text = scrolledtext.ScrolledText(self.master, height=10)
        self.output_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Entropy plot
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.ax.set_title('STFT(Attention Entropy)')
        self.ax.set_xlabel('Samples')
        self.ax.set_ylabel('Entropy')
        self.line, = self.ax.plot([], [], label='Entropy')
        self.ax.legend()

    def start_generation(self):
        if not self.is_generating:
            self.is_generating = True
            self.stop_generation = False
            self.generate_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.entropy_values = []
            prompt = self.input_text.get(1.0, tk.END).strip()

            # Start the generation in a new thread
            self.generation_thread = threading.Thread(target=self.generate_text, args=(prompt,))
            self.generation_thread.start()

    def stop_generation_flag(self):
        self.stop_generation = True

    def on_close(self):
        """Handle the window close event to ensure graceful shutdown."""
        self.stop_generation = True
        if self.generation_thread is not None and self.generation_thread.is_alive():
            self.generation_thread.join()
        self.master.destroy()

    def generate_text(self, prompt):
        model_params = self.model_params
        xfmr_weights = self.xfmr_weights
        tokenizer = self.tokenizer

        raw_tokens = tokenizer.encode(prompt, bos=False, eos=False)
        tokens = jnp.array([raw_tokens], jnp.int32)
        seqlen = tokens.shape[1]
        cur_pos = 0
        # Build the initial attention mask
        attn_mask = build_attn_mask(seqlen, cur_pos)

        freqs_cis = precompute_freqs_cis(
            dim=model_params.head_dim,
            end=model_params.max_seq_len,
            theta=model_params.rope_theta,
            use_scaled=model_params.use_scaled
        )
        kvcache = KVCache.new(
            layers=model_params.n_layers,
            bsz=tokens.shape[0],
            max_seq_len=model_params.max_seq_len,
            kv_heads=model_params.n_local_kv_heads,
            head_dim=model_params.head_dim
        )

        logits, kvcache, attn_stats = xfmr(
            xfmr_weights=xfmr_weights,
            model_params=model_params,
            tokens=tokens,
            cur_pos=0,
            freqs_cis=freqs_cis[:seqlen],
            kvcache=kvcache,
            attn_mask=attn_mask
        )
        next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
        cur_pos = seqlen
        stop = jnp.array([128001, 128008, 128009])

        while cur_pos < 2048 and not self.stop_generation:
            cur_pos += 1
            # Build the attention mask for the new sequence length
            logits, kvcache, attn_stats = xfmr(
                xfmr_weights=xfmr_weights,
                model_params=model_params,
                tokens=next_token,
                cur_pos=cur_pos,
                freqs_cis=freqs_cis[cur_pos:cur_pos+1],
                kvcache=kvcache
            )

            # Append all entropy values for this step
            self.entropy_values += attn_stats.avg_entropy[0].tolist()
            next_token = sample(logits)
            token_text = tokenizer.decode(next_token.tolist()[0])

            self.master.after(0, self.update_ui, token_text)

            if jnp.isin(next_token, stop).any():
                break

        self.master.after(0, self.generation_complete)

    def update_ui(self, token_text):
        """Update the GUI elements in the main thread."""
        # Update the output text
        self.output_text.insert(tk.END, token_text)
        self.output_text.see(tk.END)

        # Update the entropy plot
        self.update_plot()

    def generation_complete(self):
        self.is_generating = False
        self.generate_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # Start of Selection
    def update_plot(self):
            """Update the GUI elements in the main thread with STFT-transformed entropy and raw entropy."""
            # Check if there are entropy values to plot
            if not self.entropy_values:
                return

            # Convert list to a numpy array
            entropy_array = np.array(self.entropy_values)

            window_size = self.model_params.n_layers * 8
            step_size = 1    # Define step size for sliding window

            if len(entropy_array) < window_size:
                print("Not enough data for STFT with the given window size.")
                return

            # Compute Short-Time Fourier Transform (STFT) with proper windowing
            stft_magnitudes = []
            window_fn = np.hamming(window_size)
            for i in range(0, len(entropy_array) - window_size + 1, step_size):
                window = entropy_array[i:i + window_size]
                windowed = window * window_fn
                fft_result = np.fft.fft(windowed)
                magnitudes = np.abs(fft_result)
                # Aggregate magnitudes (e.g., take the mean of the magnitudes)
                mean_magnitude = np.mean(magnitudes)
                stft_magnitudes.append(mean_magnitude)

            stft_magnitudes = np.array(stft_magnitudes)

            # x-values for raw entropy
            x_values_entropy = np.arange(len(entropy_array))

            # x-values for STFT magnitudes (center of each window)
            x_values_stft = np.arange(len(stft_magnitudes)) + window_size // 2

            # Clear the previous plot
            self.ax.clear()

            # Plot the raw entropy values
            self.ax.plot(x_values_entropy, entropy_array, label='Raw Entropy')

            # Plot the STFT magnitudes
            self.ax.plot(x_values_stft, stft_magnitudes, label='STFT Magnitude')

            self.ax.set_title('STFT(Attention Entropy) and Raw Entropy Over Time')
            self.ax.set_xlabel('Ticks')
            self.ax.set_ylabel('Entropy')
            self.ax.legend()

            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
            self.canvas.flush_events()

if __name__ == "__main__":
    root = tk.Tk()
    app = TextCompletionUI(root)
    root.mainloop()