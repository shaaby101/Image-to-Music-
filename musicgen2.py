"""
image_music_fft.py
==================
Converts an image to music using FFT-based synthesis and rich image property mapping.

Image Properties Mapped:
  - Brightness      → melodic register (octave / note height)
  - Contrast        → harmonic richness (number of FFT overtones)
  - Saturation      → reverb / echo intensity
  - Hue             → scale selection (warm=major, cool=minor/exotic)
  - Edge density    → rhythmic density (more edges = faster notes)
  - Color variance  → waveform distortion amount
  - Red channel     → bass voice frequency content
  - Green channel   → melody voice frequency content
  - Blue channel    → percussion timbre
  - Local entropy   → note duration randomness

Core synthesis:
  - All tones are built with Inverse FFT (IFFT) instead of direct sin/cos formulas.
  - Harmonic overtones are added in the frequency domain before IFFT.
  - Percussion uses FFT-shaped noise bursts.
"""

import numpy as np
from PIL import Image
from scipy.io.wavfile import write
from scipy.fft import fft, ifft, fftfreq
import random
import colorsys

# ---------------------------------------------------------------------------
# SCALES  (12-TET frequencies, one octave each)
# ---------------------------------------------------------------------------
SCALES = {
    "major":       [1.000, 1.122, 1.260, 1.335, 1.498, 1.682, 1.888, 2.000],
    "minor":       [1.000, 1.122, 1.189, 1.335, 1.498, 1.587, 1.782, 2.000],
    "pentatonic":  [1.000, 1.122, 1.260, 1.498, 1.682, 2.000],
    "phrygian":    [1.000, 1.059, 1.189, 1.335, 1.498, 1.587, 1.782, 2.000],
    "lydian":      [1.000, 1.122, 1.260, 1.414, 1.498, 1.682, 1.888, 2.000],
    "blues":       [1.000, 1.189, 1.335, 1.414, 1.498, 1.782, 2.000],
}

# Base root notes per register (brightness drives octave selection)
ROOT_REGISTERS = [65.41, 130.81, 261.63, 523.25, 1046.50]  # C2 → C6


# ---------------------------------------------------------------------------
# IMAGE ANALYSIS  — extract all musical parameters from one image
# ---------------------------------------------------------------------------

def analyze_image(image_path):
    """
    Opens the image and extracts a flat dictionary of normalized [0,1]
    musical parameters.  Every downstream function reads from this dict.
    """
    img_rgb = Image.open(image_path).convert("RGB").resize((64, 64))
    arr = np.array(img_rgb, dtype=np.float32) / 255.0      # shape (64,64,3)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # ---- per-pixel HSV conversion ----------------------------------------
    hsv = np.array([
        colorsys.rgb_to_hsv(arr[y, x, 0], arr[y, x, 1], arr[y, x, 2])
        for y in range(64) for x in range(64)
    ]).reshape(64, 64, 3)

    hue_map   = hsv[:, :, 0]   # 0–1  (0=red, 0.33=green, 0.66=blue, 1=red)
    sat_map   = hsv[:, :, 1]   # 0–1
    val_map   = hsv[:, :, 2]   # 0–1  (same as brightness for pure colours)

    # ---- scalar image statistics -----------------------------------------
    brightness   = float(np.mean(val_map))          # 0–1
    contrast     = float(np.std(val_map))            # 0–0.5 approx
    saturation   = float(np.mean(sat_map))           # 0–1
    mean_hue     = float(np.mean(hue_map))           # 0–1
    color_var    = float(np.std(arr))                # 0–1 approx

    # ---- edge density via Sobel-like gradient ----------------------------
    gray = 0.299*r + 0.587*g + 0.114*b
    gx   = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy   = np.diff(gray, axis=0, prepend=gray[:1, :])
    edge_density = float(np.mean(np.sqrt(gx**2 + gy**2)))   # 0–1 approx

    # ---- local entropy proxy (variance in 4x4 blocks) -------------------
    block_vars = []
    for y in range(0, 64, 4):
        for x in range(0, 64, 4):
            block_vars.append(float(np.var(gray[y:y+4, x:x+4])))
    entropy_score = float(np.mean(block_vars))      # 0–1 approx, normalize below

    # ---- FFT of each colour channel row-wise (spectral energy) ----------
    r_fft_energy = float(np.mean(np.abs(fft(r, axis=1))[:, 1:32]))
    g_fft_energy = float(np.mean(np.abs(fft(g, axis=1))[:, 1:32]))
    b_fft_energy = float(np.mean(np.abs(fft(b, axis=1))[:, 1:32]))

    # ---- normalise to [0,1] with sensible clamps ------------------------
    def clamp(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v))

    params = {
        # Raw channel arrays (used later for per-column note picking)
        "r_channel": r,
        "g_channel": g,
        "b_channel": b,
        "gray":      gray,
        "hue_map":   hue_map,
        "sat_map":   sat_map,

        # Scalar musical parameters
        "brightness":    clamp(brightness),
        "contrast":      clamp(contrast * 3.0),          # stretch ~0–0.5 → 0–1
        "saturation":    clamp(saturation),
        "mean_hue":      clamp(mean_hue),
        "color_var":     clamp(color_var * 3.0),
        "edge_density":  clamp(edge_density * 8.0),      # typically very small
        "entropy":       clamp(entropy_score * 50.0),    # block variance is tiny
        "r_energy":      clamp(r_fft_energy / 30.0),
        "g_energy":      clamp(g_fft_energy / 30.0),
        "b_energy":      clamp(b_fft_energy / 30.0),
    }

    print("\n=== Image Analysis ===")
    for k, v in params.items():
        if not isinstance(v, np.ndarray):
            print(f"  {k:<18} {v:.4f}")

    return params


# ---------------------------------------------------------------------------
# SCALE SELECTION  — hue decides the mood / scale
# ---------------------------------------------------------------------------

def choose_scale(mean_hue):
    """
    Warm hues (red/orange/yellow)  → bright scales (major, lydian)
    Cool hues (blue/cyan/purple)   → dark scales (minor, phrygian)
    Green / balanced               → pentatonic / blues
    """
    if mean_hue < 0.10 or mean_hue > 0.90:   # red
        return "major"
    elif mean_hue < 0.20:                      # orange
        return "lydian"
    elif mean_hue < 0.40:                      # yellow-green
        return "pentatonic"
    elif mean_hue < 0.55:                      # green
        return "blues"
    elif mean_hue < 0.72:                      # blue
        return "phrygian"
    else:                                      # purple/violet
        return "minor"


# ---------------------------------------------------------------------------
# FFT-BASED TONE SYNTHESIS
# ---------------------------------------------------------------------------

def fft_tone(freq, duration, num_harmonics, harmonic_rolloff=0.6, sr=44100):
    """
    Builds a tone entirely in the frequency domain, then converts to audio
    via IFFT.  This is the core FFT synthesis engine.

    Parameters
    ----------
    freq           : fundamental frequency in Hz
    duration       : note length in seconds
    num_harmonics  : how many overtones to add (contrast-driven)
    harmonic_rolloff: amplitude decay per overtone (color_var-driven)
    sr             : sample rate
    """
    n = int(sr * duration)
    if n == 0:
        return np.zeros(1, dtype=np.float32)

    # Frequency resolution of this buffer
    freq_resolution = sr / n          # Hz per FFT bin
    spectrum = np.zeros(n, dtype=complex)

    for h in range(1, num_harmonics + 1):
        harmonic_freq = freq * h
        if harmonic_freq >= sr / 2:   # above Nyquist — skip
            break

        # Which FFT bin does this frequency fall into?
        bin_idx = int(round(harmonic_freq / freq_resolution))
        if bin_idx >= n:
            break

        # Amplitude: fundamental=1, each overtone scaled by rolloff^(h-1)
        amplitude = (harmonic_rolloff ** (h - 1)) * (n / 2)

        # Add a small random phase for naturalness (avoids a pure square-wave feel)
        phase = random.uniform(0, 2 * np.pi)

        # Set positive and mirror negative frequency bin
        spectrum[bin_idx]     += amplitude * np.exp(1j * phase)
        spectrum[n - bin_idx] += amplitude * np.exp(-1j * phase)  # conjugate mirror

    # Convert frequency domain → time domain
    audio = np.real(ifft(spectrum)).astype(np.float32)

    # Normalise to ±1
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    return audio


def fft_noise_burst(duration, spectral_shape, sr=44100):
    """
    Percussion synthesis: white noise shaped by a spectral envelope in the
    frequency domain, then IFFT'd back.

    spectral_shape: 1D array of length (n//2+1) containing desired amplitudes
                    per frequency bin — controls the colour of the noise.
    """
    n = int(sr * duration)
    if n == 0:
        return np.zeros(1, dtype=np.float32)

    # White noise in frequency domain (random amplitude + random phase)
    freqs_count = n // 2 + 1
    shape_len   = len(spectral_shape)

    # Resample spectral_shape to match buffer size
    indices     = (np.arange(freqs_count) * (shape_len - 1) / max(freqs_count - 1, 1)).astype(int)
    indices     = np.clip(indices, 0, shape_len - 1)
    envelope    = spectral_shape[indices]

    phases      = np.random.uniform(0, 2 * np.pi, freqs_count)
    pos_freqs   = envelope * np.exp(1j * phases)

    # Build full conjugate-symmetric spectrum for real-valued output
    full_spectrum           = np.zeros(n, dtype=complex)
    full_spectrum[:freqs_count] = pos_freqs
    if n % 2 == 0:
        full_spectrum[freqs_count:] = np.conj(pos_freqs[-2:0:-1])
    else:
        full_spectrum[freqs_count:] = np.conj(pos_freqs[-1:0:-1])

    audio = np.real(ifft(full_spectrum)).astype(np.float32)
    peak  = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


# ---------------------------------------------------------------------------
# SPECTRAL SHAPES FOR PERCUSSION  (driven by blue channel energy)
# ---------------------------------------------------------------------------

def make_kick_shape(n, low_emphasis=0.8):
    """
    Strong low-frequency energy, rapid high-frequency rolloff.
    low_emphasis: how much to boost lows (b_energy-driven)
    """
    shape = np.zeros(n)
    for i in range(n):
        # Exponential falloff from bin 0; low_emphasis boosts bass
        shape[i] = np.exp(-i * (1.0 - low_emphasis * 0.9) / max(n * 0.05, 1))
    return shape


def make_snare_shape(n, mid_boost=0.5):
    """
    Noise with a mid-frequency bump.
    """
    shape = np.ones(n) * 0.3
    mid   = int(n * 0.3)
    width = int(n * 0.2)
    for i in range(max(0, mid - width), min(n, mid + width)):
        shape[i] += mid_boost * np.exp(-((i - mid)**2) / (2 * (width / 2)**2))
    return shape


def make_hihat_shape(n):
    """
    High-frequency emphasis only.
    """
    shape = np.zeros(n)
    cutoff = int(n * 0.7)
    shape[cutoff:] = np.linspace(0, 1, n - cutoff)
    return shape


# ---------------------------------------------------------------------------
# ADSR ENVELOPE  (unchanged logic, still needed after IFFT)
# ---------------------------------------------------------------------------

def apply_adsr(signal, sr=44100, attack=0.01, decay=0.05, sustain=0.7, release=0.1):
    n = len(signal)
    if n == 0:
        return signal
    env = np.ones(n, dtype=np.float32)
    a   = min(int(attack  * sr), n)
    d   = min(int(decay   * sr), n - a)
    r   = min(int(release * sr), n - a - d)
    if a > 0: env[:a]        = np.linspace(0, 1,       a)
    if d > 0: env[a:a+d]     = np.linspace(1, sustain, d)
    if r > 0: env[n - r:]    = np.linspace(sustain, 0, r)
    return signal * env


# ---------------------------------------------------------------------------
# SPECTRAL REVERB  (convolution-based; saturation controls wet mix)
# ---------------------------------------------------------------------------

def spectral_reverb(signal, room_size=0.4, wet=0.3, sr=44100):
    """
    Reverb by multiplying the signal's FFT by a decaying noise spectrum,
    then IFFT.  Completely FFT-based — no delay lines.

    room_size: controls decay length of the impulse response
    wet:       mix ratio (saturation-driven)
    """
    if wet < 0.01:
        return signal

    n  = len(signal)
    ir_len = max(int(sr * room_size), 1)

    # Build a simple exponential-decay impulse response
    t_ir = np.linspace(0, room_size, ir_len)
    ir   = np.random.randn(ir_len).astype(np.float32) * np.exp(-6 * t_ir / room_size)

    # Zero-pad both to the same length for linear convolution
    fft_len   = n + ir_len - 1
    sig_fft   = fft(signal,  fft_len)
    ir_fft    = fft(ir,      fft_len)
    reverbed  = np.real(ifft(sig_fft * ir_fft)).astype(np.float32)

    # Trim to original length
    reverbed  = reverbed[:n]

    # Normalise the wet signal
    peak = np.max(np.abs(reverbed))
    if peak > 0:
        reverbed /= peak

    return (1 - wet) * signal + wet * reverbed


# ---------------------------------------------------------------------------
# NOTE PICKING  — brightness of an image column → frequency
# ---------------------------------------------------------------------------

def pick_freq_from_column(channel, col_idx, scale_ratios, root_freq):
    """
    Averages the pixel values in a single column of a channel,
    maps 0–1 brightness to a scale degree, returns frequency in Hz.
    """
    col_brightness = float(np.mean(channel[:, col_idx]))   # 0–1 (already normalised)
    degree         = int(col_brightness * (len(scale_ratios) - 1))
    return root_freq * scale_ratios[degree]


# ---------------------------------------------------------------------------
# VOICE GENERATORS
# ---------------------------------------------------------------------------

def generate_melody(params, scale_ratios, root_freq, total_duration=10, sr=44100):
    """
    Melody voice: green channel → frequency, contrast → harmonics,
    color_var → harmonic rolloff, edge_density → note density.
    """
    channel       = params["g_channel"]
    num_harmonics = max(1, int(params["contrast"] * 8) + 1)    # 1–9 overtones
    rolloff       = 0.3 + params["color_var"] * 0.6            # 0.3–0.9
    width         = channel.shape[1]

    # Edge density: high edges → shorter notes (busier rhythm)
    short_dur  = 0.125
    long_dur   = 0.5
    edge       = params["edge_density"]                          # 0–1
    dur_pool   = [short_dur] * max(1, int(edge * 6)) + \
                 [0.25]      * 3 + \
                 [long_dur]  * max(1, int((1 - edge) * 4))

    notes, current_time = [], 0.0
    while current_time < total_duration:
        col      = random.randint(0, width - 1)
        duration = random.choice(dur_pool)
        freq     = pick_freq_from_column(channel, col, scale_ratios, root_freq)

        note = fft_tone(freq, duration, num_harmonics, rolloff, sr)
        note = apply_adsr(note, sr, attack=0.02, decay=0.04, sustain=0.75, release=0.12)
        notes.append(note)
        current_time += duration

    return np.concatenate(notes).astype(np.float32) if notes else np.zeros(int(total_duration * sr), dtype=np.float32)


def generate_bass(params, scale_ratios, root_freq, total_duration=10, sr=44100):
    """
    Bass voice: red channel → frequency (one octave lower),
    r_energy → harmonic richness (more energy = more overtones).
    Bass always uses slow rhythm (long notes).
    """
    channel       = params["r_channel"]
    bass_root     = root_freq / 2.0                             # one octave down
    num_harmonics = max(1, int(params["r_energy"] * 6) + 1)    # 1–7
    rolloff       = 0.5 + params["color_var"] * 0.4
    width         = channel.shape[1]

    dur_pool      = [0.5, 0.5, 1.0]                            # bass is slow

    notes, current_time = [], 0.0
    while current_time < total_duration:
        col      = random.randint(0, width - 1)
        duration = random.choice(dur_pool)
        freq     = pick_freq_from_column(channel, col, scale_ratios, bass_root)

        note = fft_tone(freq, duration, num_harmonics, rolloff, sr)
        note = apply_adsr(note, sr, attack=0.05, decay=0.08, sustain=0.6, release=0.2)
        notes.append(note)
        current_time += duration

    return np.concatenate(notes).astype(np.float32) if notes else np.zeros(int(total_duration * sr), dtype=np.float32)


def generate_pad(params, scale_ratios, root_freq, total_duration=10, sr=44100):
    """
    Atmospheric pad voice: saturation → intensity, entropy → harmonic movement.
    Uses very long notes and high harmonic count for a lush texture.
    All synthesis done with FFT tones (5–12 harmonics).
    """
    channel       = params["g_channel"]
    num_harmonics = 5 + int(params["saturation"] * 7)           # 5–12
    rolloff       = 0.7                                          # rich, dense sound
    width         = channel.shape[1]

    notes, current_time = [], 0.0
    while current_time < total_duration:
        col      = random.randint(0, width - 1)
        duration = random.choice([1.0, 2.0])                    # long sustained notes
        freq     = pick_freq_from_column(channel, col, scale_ratios, root_freq * 0.75)

        note = fft_tone(freq, duration, num_harmonics, rolloff, sr)
        note = apply_adsr(note, sr, attack=0.3, decay=0.1, sustain=0.8, release=0.4)
        notes.append(note)
        current_time += duration

    raw = np.concatenate(notes).astype(np.float32) if notes else np.zeros(int(total_duration * sr), dtype=np.float32)
    return raw * 0.4   # pads are quiet background


def generate_percussion(params, total_duration=10, sr=44100):
    """
    Percussion: blue channel energy → spectral shape of each drum.
    Edge density → hit density (more edges = more frequent hits).
    """
    b_energy    = params["b_energy"]
    edge        = params["edge_density"]

    # Spectral shapes (short buffer for shape design, resampled in fft_noise_burst)
    shape_n    = 128
    kick_shape  = make_kick_shape (shape_n, low_emphasis=0.4 + b_energy * 0.5)
    snare_shape = make_snare_shape(shape_n, mid_boost=0.3 + b_energy * 0.4)
    hihat_shape = make_hihat_shape(shape_n)

    # Silence probability: high edge → less silence → busier percussion
    silence_prob = max(0.1, 0.6 - edge * 0.5)
    drum_pool    = [
        (drum_kick_fft,  kick_shape,  0.2),
        (drum_snare_fft, snare_shape, 0.2),
        (drum_hihat_fft, hihat_shape, 0.2),
        (None,           None,        silence_prob),
    ]
    # Normalise weights
    total_w  = sum(w for _, _, w in drum_pool)
    weights  = [w / total_w for _, _, w in drum_pool]
    funcs    = [(f, s) for f, s, _ in drum_pool]

    pattern, current_time = [], 0.0
    while current_time < total_duration:
        choice_idx  = random.choices(range(len(funcs)), weights=weights, k=1)[0]
        func, shape = funcs[choice_idx]
        duration    = random.choice([0.125, 0.25])

        if func is not None:
            hit = func(shape, duration, sr)
            decay_env = np.exp(-15 * np.linspace(0, duration, len(hit)))
            hit = (hit * decay_env).astype(np.float32)
            pattern.append(hit)
        else:
            pattern.append(np.zeros(int(sr * duration), dtype=np.float32))

        current_time += duration

    return np.concatenate(pattern).astype(np.float32) if pattern else np.zeros(int(total_duration * sr), dtype=np.float32)


# Thin wrappers so generate_percussion can call them uniformly
def drum_kick_fft (shape, duration, sr): return fft_noise_burst(duration, shape, sr)
def drum_snare_fft(shape, duration, sr): return fft_noise_burst(duration, shape, sr)
def drum_hihat_fft(shape, duration, sr): return fft_noise_burst(duration, shape, sr)


# ---------------------------------------------------------------------------
# MIXING & MASTERING
# ---------------------------------------------------------------------------

def mix_and_master(voices, params, sr=44100):
    """
    1. Apply per-voice spectral reverb (wet amount = saturation).
    2. Sum all voices with level weights.
    3. Normalise.
    4. Spectral soft-clip in frequency domain (colour_var-driven).
    """
    wet        = params["saturation"] * 0.6          # 0–0.6 reverb mix
    room_size  = 0.2 + params["entropy"] * 0.5       # 0.2–0.7 s

    level_weights = {"melody": 1.0, "bass": 0.7, "pad": 0.5, "percussion": 0.6}

    max_len = max(len(v) for v in voices.values())
    mix     = np.zeros(max_len, dtype=np.float32)

    for name, voice in voices.items():
        reverbed  = spectral_reverb(voice, room_size=room_size, wet=wet, sr=sr)
        weight    = level_weights.get(name, 1.0)
        mix[:len(reverbed)] += reverbed * weight

    # ---- FFT soft-clip (frequency-domain saturation) --------------------
    # Drives harmonic saturation from color_var; more varied colour = warmer master
    distortion = params["color_var"] * 0.4           # 0–0.4

    if distortion > 0.01:
        n          = len(mix)
        spectrum   = fft(mix)
        magnitudes = np.abs(spectrum)
        phases     = np.angle(spectrum)

        # Soft-clip magnitudes: tanh(x) never exceeds 1 — smoothly limits peaks
        peak_mag   = np.max(magnitudes)
        if peak_mag > 0:
            norm_mags  = magnitudes / peak_mag
            clipped    = np.tanh(norm_mags * (1 + distortion * 3))
            spectrum   = clipped * peak_mag * np.exp(1j * phases)
            mix        = np.real(ifft(spectrum)).astype(np.float32)

    # ---- Final normalise -------------------------------------------------
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix /= peak

    return np.clip(mix, -1.0, 1.0)


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def generate_image_music(
    image_path,
    output_wav      = "output.wav",
    total_duration  = 10,
    sr              = 44100,
):
    """
    Full pipeline:
      1. Analyse image → extract musical parameters
      2. Choose scale from hue
      3. Choose root octave from brightness
      4. Generate melody, bass, pad, percussion voices
      5. Mix & master with spectral reverb + FFT soft-clip
      6. Write WAV
    """
    # ---- 1. Analysis ------------------------------------------------------
    params = analyze_image(image_path)

    # ---- 2. Random seed deterministic from image -------------------------
    seed = int(np.sum(np.array(Image.open(image_path))) % 2**32)
    random.seed(seed)
    np.random.seed(seed)

    # ---- 3. Scale & root -------------------------------------------------
    scale_name   = choose_scale(params["mean_hue"])
    scale_ratios = SCALES[scale_name]

    # Brightness maps to octave register
    register_idx = int(params["brightness"] * (len(ROOT_REGISTERS) - 1))
    root_freq    = ROOT_REGISTERS[register_idx]

    print(f"\n=== Music Parameters ===")
    print(f"  Scale          : {scale_name}")
    print(f"  Root frequency : {root_freq:.2f} Hz  (register {register_idx})")
    print(f"  Num harmonics  : {max(1, int(params['contrast']*8)+1)}")
    print(f"  Reverb wet     : {params['saturation']*0.6:.2f}")
    print(f"  Reverb room    : {0.2 + params['entropy']*0.5:.2f} s")
    print(f"  FFT saturation : {params['color_var']*0.4:.2f}")

    # ---- 4. Generate voices ----------------------------------------------
    print("\nGenerating melody ...")
    melody     = generate_melody    (params, scale_ratios, root_freq, total_duration, sr)

    print("Generating bass ...")
    bass       = generate_bass      (params, scale_ratios, root_freq, total_duration, sr)

    print("Generating pad ...")
    pad        = generate_pad       (params, scale_ratios, root_freq, total_duration, sr)

    print("Generating percussion ...")
    percussion = generate_percussion(params,                           total_duration, sr)

    voices = {
        "melody":     melody,
        "bass":       bass,
        "pad":        pad,
        "percussion": percussion,
    }

    # ---- 5. Mix & master --------------------------------------------------
    print("Mixing & mastering ...")
    final_mix = mix_and_master(voices, params, sr)

    # ---- 6. Write WAV -----------------------------------------------------
    write(output_wav, sr, (final_mix * 32767).astype(np.int16))
    print(f"\n✓ Saved: {output_wav}")

    return final_mix, voices, sr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python image_music_fft.py <image_path> <output.wav> [duration_seconds]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_wav = sys.argv[2]
    duration   = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    generate_image_music(image_path, output_wav, total_duration=duration)