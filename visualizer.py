"""
visualizer.py
=============
Static spectrogram generator for Image-to-Music compositions.

This module generates high-quality PNG spectrograms showing the frequency
content analysis of the generated mixed audio signal using FFT.

Usage:
    from musicgen2 import generate_image_music
    from visualizer import generate_spectrogram

    mix, voices, sr = generate_image_music('image.jpg', 'output.wav', 10)
    spectrogram_path = generate_spectrogram(mix, sr, 'spectrogram.png')
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm


def generate_spectrogram(audio_signal, sample_rate, output_path, figsize=(12, 6), dpi=150):
    """
    Generates a static spectrogram image from the mixed audio signal.

    A spectrogram shows the frequency content of the audio over time, with:
    - X-axis: Time (seconds)
    - Y-axis: Frequency (Hz)
    - Color intensity: Power (magnitude in dB)

    The visualization uses FFT with a 1024-point window and 512-point overlap
    for detailed time-frequency resolution.

    Parameters
    ----------
    audio_signal : np.ndarray
        1D numpy array of float32 audio samples (the final mixed signal).
    sample_rate : int
        Audio sample rate in Hz (typically 44100).
    output_path : str
        Output PNG file path where spectrogram will be saved.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (12, 6).
    dpi : int, optional
        Resolution in dots per inch. Default: 150 (high quality).

    Returns
    -------
    str
        Path to the saved PNG file.

    Notes
    -----
    - Uses FFT for frequency analysis
    - Dynamic range displayed: -80 to 0 dB
    - Uses "magma" colormap (perceptually uniform, publication-ready)
    - Includes colorbar, gridlines, and axis labels
    """

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Generate spectrogram (returns power in dB)
    # NFFT=1024: Window size for FFT (frequency resolution)
    # noverlap=512: 50% overlap between windows (time resolution)
    Pxx, freqs, bins, im = ax.specgram(
        audio_signal,
        Fs=sample_rate,
        NFFT=1024,
        noverlap=512,
        cmap="magma",
        norm=PowerNorm(gamma=0.3),  # Power normalization for better contrast
        vmin=-80,
        vmax=0
    )

    # Formatting
    ax.set_title("📊 Image-to-Music: Mixed Signal Spectrogram (FFT Analysis)", 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_ylim([0, sample_rate // 2])  # Show up to Nyquist frequency

    # Colorbar (shows power in dB)
    cbar = plt.colorbar(im, ax=ax, label="Power (dB)")

    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return output_path


# --- COMMAND-LINE USAGE ---
if __name__ == "__main__":
    import sys

    print(__doc__)
    
    print("\n=== Streamlit Integration ===")
    print("This spectrogram generator is automatically called by st_app.py")
    print("The PNG image is displayed in the Streamlit UI via st.image()")
    
    print("\n=== Direct Python Usage ===")
    print("from musicgen2 import generate_image_music")
    print("from visualizer import generate_spectrogram")
    print("")
    print("# Generate music from image")
    print("mix, voices, sr = generate_image_music('image.jpg', 'output.wav', duration=10)")
    print("")
    print("# Create spectrogram visualization")
    print("spectrogram_path = generate_spectrogram(mix, sr, 'spectrogram.png')")
    print("")
    print("# Result: spectrogram.png contains frequency analysis of the audio")


