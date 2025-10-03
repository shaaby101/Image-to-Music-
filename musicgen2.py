import numpy as np
from PIL import Image
from scipy.io.wavfile import write
import random

# --- Scales (full notes, not just pentatonic) ---
SCALES = {
    "C_major": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25],
    "A_minor": [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.00]
}

# --- Waveforms ---
def sine_wave(freq, duration, sr=44100):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    return np.sin(2*np.pi*freq*t)

def square_wave(freq, duration, sr=44100):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    return np.sign(np.sin(2*np.pi*freq*t))

# --- Percussion (drum patterns) ---
def drum_kick(duration, sr=44100):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    return np.sin(2*np.pi*60*t)*np.exp(-10*t)

def drum_snare(duration, sr=44100):
    return np.random.uniform(-1,1,int(sr*duration))*np.exp(-15*np.linspace(0,duration,int(sr*duration)))

def drum_hihat(duration, sr=44100):
    return np.random.uniform(-0.3,0.3,int(sr*duration))

# --- Envelopes ---
def apply_adsr(signal, sr=44100, attack=0.01, decay=0.05, sustain=0.7, release=0.1):
    n = len(signal)
    if n==0: return signal
    env = np.ones(n)
    a = min(int(attack*sr), n)
    d = min(int(decay*sr), n-a)
    r = min(int(release*sr), n-a-d)
    if a>0: env[:a] = np.linspace(0,1,a)
    if d>0: env[a:a+d] = np.linspace(1,sustain,d)
    if r>0: env[n-r:] = np.linspace(sustain,0,r)
    return signal*env

# --- Effects ---
def add_echo(signal, delay=0.2, decay=0.4, sr=44100):
    d = int(sr*delay)
    out = np.copy(signal)
    for i in range(d,len(signal)):
        out[i] += decay*out[i-d]
    return out

# --- Map image to notes ---
def pick_note_from_image(img_array, scale_freqs, idx):
    # brightness -> note index
    brightness = np.mean(img_array[:, idx])/255.0 if img_array.ndim==2 else np.mean(img_array[idx])/255.0
    note_idx = int(brightness*(len(scale_freqs)-1))
    return scale_freqs[note_idx]

# --- Generate melody/bass/percussion voices ---
def generate_voice(img_channel, scale_freqs, waveform_func, total_duration=10, sr=44100):
    notes = []
    current_time = 0.0
    width = img_channel.shape[1] if img_channel.ndim==2 else img_channel.shape[0]
    while current_time < total_duration:
        idx = random.randint(0,width-1)
        duration = random.choice([0.125,0.25,0.5])
        freq = pick_note_from_image(img_channel, scale_freqs, idx)
        note = waveform_func(freq,duration,sr)
        note = apply_adsr(note,sr)
        notes.append(note)
        current_time += duration
    if not notes: return np.zeros(int(total_duration*sr),dtype=np.float32)
    return np.concatenate(notes).astype(np.float32)

def generate_percussion(img_channel, total_duration=10, sr=44100):
    pattern = []
    current_time=0.0
    width = img_channel.shape[1] if img_channel.ndim==2 else img_channel.shape[0]
    while current_time < total_duration:
        idx = random.randint(0,width-1)
        choice = random.choices(
            [drum_kick, drum_snare, drum_hihat, None],
            weights=[0.2,0.2,0.2,0.4], k=1
        )[0]
        duration = random.choice([0.125,0.25])
        if choice:
            note = choice(duration,sr)
            pattern.append(note)
        current_time += duration
    if not pattern: return np.zeros(int(total_duration*sr),dtype=np.float32)
    return np.concatenate(pattern).astype(np.float32)

# --- Mix voices ---
def mix_voices(voices):
    max_len = max(len(v) for v in voices.values())
    mix = np.zeros(max_len,dtype=np.float32)
    for v in voices.values():
        mix[:len(v)] += v
    mx = np.max(np.abs(mix))
    if mx>0: mix = mix/mx
    return mix

def generate_percussion(img_channel, total_duration=10, sr=44100, bpm=120):
    # 16 steps per bar
    step_time = 60 / bpm / 4  # quarter note divided into 4 steps
    num_steps = int(total_duration / step_time)
    pattern = []

    for i in range(num_steps):
        # Basic drum pattern probabilities
        kick = drum_kick(step_time, sr) if i % 4 == 0 else None  # kick on 1 & 3
        snare = drum_snare(step_time, sr) if i % 4 == 2 else None # snare on 2 & 4
        hihat = drum_hihat(step_time, sr) if i % 1 == 0 else None # hi-hat every step

        step_audio = np.zeros(int(step_time*sr))
        for drum in [kick, snare, hihat]:
            if drum is not None:
                step_audio += drum
        pattern.append(step_audio)

    return np.concatenate(pattern).astype(np.float32)

# --- Main generator ---
def generate_image_music(image_path, output_wav, scale_name="C_major",
                         total_duration=10, sr=44100, save_visualizer=False,
                         visualizer_func=None, vis_output=None):
    # Random seed based on image for uniqueness
    img_seed = int(np.sum(np.array(Image.open(image_path)))%2**32)
    random.seed(img_seed)
    np.random.seed(img_seed)

    img = Image.open(image_path).convert('RGB').resize((64,64))
    arr = np.array(img)
    freqs = SCALES[scale_name]

    bass = generate_voice(arr[:,:,0], freqs, square_wave, total_duration, sr)
    melody = generate_voice(arr[:,:,1], freqs, sine_wave, total_duration, sr)
    percussion = generate_percussion(arr[:,:,2], total_duration, sr)

    voices = {"bass": bass, "melody": melody, "percussion": percussion}
    mix = mix_voices(voices)
    mix = add_echo(mix, delay=0.18, decay=0.35, sr=sr)
    mix = np.clip(mix, -1,1)

    write(output_wav, sr, (mix*32767).astype(np.int16))

    saved_vis = None
    if save_visualizer and visualizer_func and vis_output:
        try:
            visualizer_func(voices,sr,image_path,vis_output)
            saved_vis = vis_output
        except Exception as e:
            print("Visualizer failed:", e)
    return mix, voices, sr, output_wav, saved_vis
