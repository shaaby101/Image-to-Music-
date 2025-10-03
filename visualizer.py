
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from PIL import Image

def save_multi_voice_visualizer(voices, sample_rate, image_path, output_video_path, fps=30):
    """
    Creates an animated visualizer showing:
    - Original image (static)
    - Waveforms of each voice (bass, melody, percussion)
    - Spectrogram of the mixed signal

    Parameters:
    - voices: dict of {"bass": array, "melody": array, "percussion": array}
    - sample_rate: int
    - image_path: path to the source image
    - output_video_path: path to save .mp4
    - fps: frames per second for animation
    """
    # Combine mix for spectrogram
    max_len = max(len(v) for v in voices.values())
    mix = np.zeros(max_len, dtype=np.float32)
    for v in voices.values():
        mix[:len(v)] += v
    if np.max(np.abs(mix)) > 0:
        mix = mix / np.max(np.abs(mix))

    # Load image
    img = Image.open(image_path)

    # Create figure layout
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, figure=fig)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_bass = fig.add_subplot(gs[0, 1])
    ax_melody = fig.add_subplot(gs[1, 0])
    ax_perc = fig.add_subplot(gs[1, 1])
    ax_spec = fig.add_subplot(gs[2, :])

    # Display static image
    ax_img.imshow(img)
    ax_img.set_title("Source Image")
    ax_img.axis("off")

    # Prepare waveform plots
    t_bass = np.arange(len(voices["bass"])) / sample_rate
    t_melody = np.arange(len(voices["melody"])) / sample_rate
    t_perc = np.arange(len(voices["percussion"])) / sample_rate
    t_mix = np.arange(len(mix)) / sample_rate

    line_bass, = ax_bass.plot([], [], lw=1, color='red')
    ax_bass.set_xlim(0, len(voices["bass"]) / sample_rate)
    ax_bass.set_ylim(-1, 1)
    ax_bass.set_title("Bass (Red channel)")

    line_melody, = ax_melody.plot([], [], lw=1, color='green')
    ax_melody.set_xlim(0, len(voices["melody"]) / sample_rate)
    ax_melody.set_ylim(-1, 1)
    ax_melody.set_title("Melody (Green channel)")

    line_perc, = ax_perc.plot([], [], lw=1, color='blue')
    ax_perc.set_xlim(0, len(voices["percussion"]) / sample_rate)
    ax_perc.set_ylim(-1, 1)
    ax_perc.set_title("Percussion (Blue channel)")

    # Spectrogram
    Pxx, freqs, bins, im = ax_spec.specgram(mix, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="magma")
    ax_spec.set_title("Mix Spectrogram")
    ax_spec.set_xlabel("Time [s]")
    ax_spec.set_ylabel("Frequency [Hz]")

    # Animation update function
    def update(frame):
        # Each frame shows a sliding window
        window = int(sample_rate * 2)  # 2 sec window
        start = frame * int(sample_rate / fps)
        end = start + window

        if end > len(mix):
            end = len(mix)

        line_bass.set_data(t_bass[:end], voices["bass"][:end])
        line_melody.set_data(t_melody[:end], voices["melody"][:end])
        line_perc.set_data(t_perc[:end], voices["percussion"][:end])

        return line_bass, line_melody, line_perc

    # Number of frames = total duration * fps
    total_duration = len(mix) / sample_rate
    frames = int(total_duration * fps)

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=1000/fps, blit=True
    )

    ani.save(output_video_path, fps=fps, dpi=150)
    plt.close(fig)

    print(f"Visualizer saved to {output_video_path}")

