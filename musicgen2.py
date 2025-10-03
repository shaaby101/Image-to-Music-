
import numpy as np
from PIL import Image
from scipy.io.wavfile import write
import random

# --- Scales ---
SCALES = {
    "C_major": {
        "freqs": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25],
        "pentatonic_idx": [0, 1, 2, 4, 5]
    },
    "A_minor": {
        "freqs": [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.00],
        "pentatonic_idx": [0, 2, 3, 4, 6]
    }
}

# --- Waveforms ---
def sine_wave(freq, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def square_wave(freq, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sign(np.sin(2 * np.pi * freq * t))

def noise_burst(duration, sample_rate=44100):
    return np.random.uniform(-1, 1, int(sample_rate * duration))

# --- Envelopes ---
def apply_adsr(signal, sample_rate=44100, attack=0.01, decay=0.05, sustain=0.7, release=0.1):
    n = len(signal)
    if n == 0:
        return signal
    env = np.ones(n)

    a = int(attack * sample_rate)
    d = int(decay * sample_rate)
    r = int(release * sample_rate)

    a = min(a, n)
    d = min(d, n - a)
    r = min(r, n - a - d)

    # Attack
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    # Decay
    if d > 0:
        env[a:a+d] = np.linspace(1, sustain, d)
    # Sustain portion remains as 'sustain'
    # Release
    if r > 0:
        env[n-r:] = np.linspace(sustain, 0, r)

    return signal * env

# --- Effects ---
def add_echo(signal, delay=0.2, decay=0.4, sample_rate=44100):
    delay_samples = int(sample_rate * delay)
    echoed = np.copy(signal)
    for i in range(delay_samples, len(signal)):
        echoed[i] += decay * signal[i - delay_samples]
    return echoed

# --- Core Music Functions ---
def pick_note_from_image(img_array, scale_freqs, idx):
    # img_array is 2D array (e.g., one channel)
    brightness = np.mean(img_array[:, idx]) / 255.0 if img_array.ndim == 2 else np.mean(img_array[idx]) / 255.0
    note_idx = int(brightness * (len(scale_freqs) - 1))
    return scale_freqs[note_idx]

def generate_voice(img_channel, scale_freqs, waveform_func, total_duration=10, sample_rate=44100):
    """
    img_channel: 2D array shape (H, W) or 1D per-column aggregated array (W,)
    We will treat input as columns: if 2D, choose brightness per column.
    """
    # Prepare a per-column brightness map
    if img_channel.ndim == 2:
        width = img_channel.shape[1]
    elif img_channel.ndim == 1:
        width = img_channel.shape[0]
    else:
        raise ValueError("img_channel must be 1D or 2D")

    notes = []
    current_time = 0.0

    # Build until we reach total_duration (may slightly overshoot)
    while current_time < total_duration:
        idx = random.randint(0, width - 1)
        duration = random.choice([0.125, 0.25, 0.5])  # varied rhythms
        freq = pick_note_from_image(img_channel, scale_freqs, idx)
        note = waveform_func(freq, duration, sample_rate)
        note = apply_adsr(note, sample_rate)
        if len(note) > 0:
            notes.append(note)
        current_time += duration

    if len(notes) == 0:
        return np.zeros(int(total_duration * sample_rate), dtype=np.float32)

    return np.concatenate(notes).astype(np.float32)

# --- Combine Voices ---
def mix_voices(voices):
    max_len = max(len(v) for v in voices.values())
    mix = np.zeros(max_len, dtype=np.float32)
    for v in voices.values():
        mix[:len(v)] += v
    # Normalize to -1..1
    max_abs = np.max(np.abs(mix))
    if max_abs > 0:
        mix = mix / max_abs
    return mix

# --- Main Generator (with optional visualizer hook) ---
def generate_image_music(
    image_path,
    output_wav,
    scale_name="C_major",
    total_duration=10,
    sample_rate=44100,
    save_visualizer=False,
    visualizer_func=None,
    vis_output=None
):
    """
    Generates voices + mix from an image.

    Parameters:
    - image_path: path to input image
    - output_wav: path to save mixed WAV
    - scale_name: one of SCALES keys
    - total_duration: seconds
    - save_visualizer: if True, call visualizer_func(voices, sample_rate, image_path, vis_output)
    - visualizer_func: callable that accepts (voices_dict, sample_rate, image_path, output_video_path)
    - vis_output: path to save visualization (required if save_visualizer True)

    Returns:
    - mix: numpy array (float32, -1..1)
    - voices: dict {"bass":..., "melody":..., "percussion":...}
    - sample_rate: int
    - output_wav: path to saved wav
    - vis_output: path to saved visualizer (or None)
    """
    # Load & normalize image to manageable size
    img = Image.open(image_path).convert('RGB').resize((64, 64))
    img_array = np.array(img)  # shape (H, W, 3)

    scale = SCALES[scale_name]
    freqs = [scale["freqs"][i] for i in scale["pentatonic_idx"]]

    # Prepare per-channel arrays; we keep them as 2D so brightness per-column is meaningful
    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    blue_channel = img_array[:, :, 2]

    # Generate voices (each returns a 1D float32 array)
    bass = generate_voice(red_channel, freqs, square_wave, total_duration, sample_rate)
    melody = generate_voice(green_channel, freqs, sine_wave, total_duration, sample_rate)
    percussion = generate_voice(blue_channel, freqs, lambda f, d, sr: noise_burst(d, sr), total_duration, sample_rate)

    voices = {"bass": bass, "melody": melody, "percussion": percussion}

    # Mix and apply effect
    mix = mix_voices(voices)
    mix = add_echo(mix, delay=0.18, decay=0.35, sample_rate=sample_rate)

    # Normalize again after effects
    max_abs = np.max(np.abs(mix))
    if max_abs > 0:
        mix = mix / max_abs

    # Save WAV
    audio_int16 = np.int16(np.clip(mix, -1.0, 1.0) * 32767)
    write(output_wav, sample_rate, audio_int16)

    # Optionally call visualizer hook
    saved_vis = None
    if save_visualizer:
        if (visualizer_func is None) or (vis_output is None):
            raise ValueError("To save a visualizer, pass visualizer_func and vis_output.")
        try:
            # visualizer_func should accept (voices_dict, sample_rate, image_path, output_video_path)
            visualizer_func(voices, sample_rate, image_path, vis_output)
            saved_vis = vis_output
        except Exception as e:
            print("Visualizer function raised an exception:", e)
            saved_vis = None

    return mix, voices, sample_rate, output_wav, saved_vis


# --- Example usage ---
if __name__ == "__main__":
    # If you have your visualizer function in a module, try importing it here.
    try:
        from visualizer import save_multi_voice_visualizer as vis_func
    except Exception:
        vis_func = None
        print("visualizer not found; pass visualizer_func manually or put your visualizer in visualizer.py")

    img_path = "example_image.jpg"  # Place your image in the script directory and update the filename if needed
    wav_path = "modular_music_mix.wav"
    vis_path = "modular_music_vis.mp4"

    mix, voices, sr, saved_wav, saved_vis = generate_image_music(
        img_path,
        wav_path,
        scale_name="C_major",
        total_duration=12,
        save_visualizer=(vis_func is not None),
        visualizer_func=vis_func,
        vis_output=vis_path if vis_func is not None else None
    )

    print("WAV saved to:", saved_wav)
    if saved_vis:
        print("Visualizer saved to:", saved_vis)
    else:
        print("Visualizer not created (visualizer function missing or errored).")

