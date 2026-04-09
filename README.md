# 🎨 Image-to-Music Generator 🎶

> **Try it now!** This project is hosted on **[Streamlit](https://image2musik.streamlit.app/)** — no installation needed. Simply visit the app and upload an image to instantly generate unique music from its visual properties!

---

## 📖 Project Overview

The **Image-to-Music Generator** is an innovative tool that transforms images into musical compositions by analyzing their visual characteristics and mapping them to musical parameters. Every aspect of the generated music—from the scale and melody to the harmonic richness and rhythm—is derived entirely from the properties of the uploaded image.

This project uses advanced **FFT-based synthesis** rather than traditional synthesizer approaches, enabling sophisticated harmonic layering and spectral processing directly in the frequency domain.

---

## 🎯 The Core Concept

### Image Properties → Musical Parameters

The algorithm extracts 10+ dimensional image analysis data and maps each visual property to a specific musical element:

| Visual Property | Musical Impact | Range |
|---|---|---|
| **Hue** | Scale selection (mood) | Red=Major, Blue=Minor, Green=Pentatonic |
| **Brightness** | Melodic register (octave) | Low=Bass, High=Treble |
| **Contrast** | Harmonic richness | Low=Simple, High=Complex |
| **Saturation** | Reverb/Echo intensity | Low=Dry, High=Wet |
| **Edge density** | Rhythmic density | Low=Sparse, High=Busy |
| **Color variance** | Waveform distortion | Low=Pure, High=Saturated |
| **Red channel** | Bass voice frequency | Controls bass note selection |
| **Green channel** | Melody voice frequency | Controls melody note selection |
| **Blue channel** | Percussion timbre | Shapes drum sound character |
| **Local entropy** | Note duration randomness | Affects rhythmic variation |


---

## 🔍 Detailed Music Generation Logic

### **Step 1: Image Analysis**

The image undergoes comprehensive analysis to extract 10+ musical parameters:

```
1. Image Resizing
   • Input image → 64×64 pixel grid
   • Normalized to [0, 1] float values

2. Color Space Conversion
   • RGB → HSV (Hue, Saturation, Value)
   • Per-pixel conversion for fine-grained analysis

3. Scalar Statistics Extraction
   • Brightness = mean of brightness values (0–1)
   • Contrast = standard deviation of brightness
   • Saturation = mean saturation across image
   • Mean Hue = average hue value (determines scale)

4. Spectral Analysis
   • Sobel-like gradient computation (edge detection)
   • Edge Density = mean gradient magnitude
   • Entropy Score = variance in 4×4 pixel blocks

5. Channel-Specific Analysis  
   • FFT applied row-wise to R, G, B channels
   • R/G/B Energy = mean spectral magnitude per channel
```


---

### **Step 2: Scale Selection (Hue → Mood)**

The mean hue value determines which musical scale powers the composition:

```python
Hue Range          Scale        Character
─────────────────────────────────────────────
0.00 – 0.10      | MAJOR      | Bright, happy (red)
0.10 – 0.20      | LYDIAN     | Dreamy, lifted (orange)
0.20 – 0.40      | PENTATONIC | Balanced, eastern (yellow–green)
0.40 – 0.55      | BLUES      | Soulful, bluesy (green)
0.55 – 0.72      | PHRYGIAN   | Dark, flamenco (blue)
0.72 – 1.00      | MINOR      | Moody, introspective (purple–violet)
```

Each scale defines the allowable note pitches:

```
MAJOR:      [C, D, E, F, G, A, B, C]     (1.0, 1.122, 1.260, 1.335, 1.498, 1.682, 1.888, 2.0)
MINOR:      [C, D, Eb, F, G, Ab, Bb, C]  (1.0, 1.122, 1.189, 1.335, 1.498, 1.587, 1.782, 2.0)
BLUES:      [C, Eb, F, F#, G, Bb, C]     (1.0, 1.189, 1.335, 1.414, 1.498, 1.782, 2.0)
PHRYGIAN:   [C, Db, Eb, F, G, Ab, Bb, C] (1.0, 1.059, 1.189, 1.335, 1.498, 1.587, 1.782, 2.0)
LYDIAN:     [C, D, E, F#, G, A, B, C]    (1.0, 1.122, 1.260, 1.414, 1.498, 1.682, 1.888, 2.0)
PENTATONIC: [C, D, E, G, A, C]           (1.0, 1.122, 1.260, 1.498, 1.682, 2.0)
```

---

### **Step 3: Root Frequency Selection (Brightness → Octave)**

Brightness determines the starting octave:

```
Brightness    Root Frequency    Note    MIDI
───────────────────────────────────────────────
0.0           65.41 Hz         C2      24
0.25          130.81 Hz        C3      36
0.5           261.63 Hz        C4      48  (middle C)
0.75          523.25 Hz        C5      60
1.0           1046.50 Hz       C6      72
```

A bright image = higher register = more "treble" composition  
A dark image = lower register = more "bass-heavy" composition


---

### **Step 4: FFT-Based Tone Synthesis**

The core innovation: **tones are synthesized entirely in the frequency domain**, not by summing sines/cosines.

#### **Traditional Approach (Sine Oscillators):**
```
Fundamental sine + N harmonic sines → Sum → Audio waveform
```

#### **FFT Approach (This Project):**
```
1. Create frequency spectrum array (size = sample count)
2. Place harmonic peaks at specific frequency bins
3. Add randomized phases to each harmonic
4. Apply Inverse FFT → Time-domain audio waveform
5. Normalize and apply ADSR envelope
```

**Why FFT synthesis?**
- ✅ Precise harmonic control in the frequency domain
- ✅ Efficient: single IFFT vs. N sine oscillators
- ✅ Enables natural spectral shaping (no aliasing artifacts)
- ✅ Seamless integration with reverb and effects (all in frequency domain)

#### **Harmonic Structure:**

```python
Number of Harmonics = 1 + int(Contrast × 8)

Harmonic #  Frequency              Amplitude
────────────────────────────────────────────────
1 (fund.)   f                      1.0
2           2f                     rolloff¹
3           3f                     rolloff²
4           4f                     rolloff³
...
N           N×f                    rolloff^(N-1)

rolloff = 0.3 + color_variance × 0.6  ∈ [0.3, 0.9]
```

- **Low contrast** → Few harmonics → Pure, bell-like tones
- **High contrast** → Many harmonics → Rich, formant-heavy tones
- **Low color variance** → Steep rolloff → Bright, thin tones
- **High color variance** → Gentle rolloff → Warm, fuzzy tones


---

### **Step 5: Multi-Voice Generation**

The system generates four independent polyphonic voices, each mapping to a different image channel:

#### **🎼 Melody Voice (Green Channel)**

```
Purpose:        Lead melodic line
Input Source:   Green channel brightness values
Frequency:      pick_freq_from_column(green_channel, col, scale, root_freq)
Harmonics:      Contrast-driven (1–9 overtones)
Rhythm:         Edge density-driven
  • High edges → short notes (0.125s) → busy rhythm
  • Low edges → long notes (0.5s) → sparse rhythm
Note Duration:  Random selection from tempo pool
ADSR Envelope:  Attack=20ms, Decay=40ms, Sustain=75%, Release=120ms
```

How it works:
1. For each randomly timed note onset:
2. Pick a random column from the image (0–63)
3. Average the green brightness in that column
4. Map brightness 0–1 → scale degree 0–7
5. Generate harmonic tone with FFT synthesis
6. Apply ADSR shaping
7. Accumulate into melody voice


---

#### **🎼 Bass Voice (Red Channel)**

```
Purpose:        Deep harmonic foundation
Input Source:   Red channel brightness values
Frequency:      pick_freq_from_column(red_channel, col, scale, root_freq/2)
                [One octave lower than root for depth]
Harmonics:      Red FFT Energy driven (1–7 overtones)
Rhythm:         Always slow (0.5s, 0.5s, 1.0s)
ADSR Envelope:  Attack=50ms, Decay=80ms, Sustain=60%, Release=200ms
                [Longer envelope for bass resonance]
```

How it works:
1. Similar to melody, but reads the RED channel instead
2. Root frequency is halved (one octave lower)
3. Harmonic count depends on red channel spectral energy
4. Rhythm is intentionally slow and steady (bass anchors the harmony)
5. Longer ADSR for sustained, resonant tones


---

#### **🎼 Pad Voice (Green Channel, Atmospheric Layer)**

```
Purpose:        Lush, sustained harmonic texture
Input Source:   Green channel (same as melody but different register)
Frequency:      pick_freq_from_column(green_channel, col, scale, root_freq × 0.75)
Harmonics:      Saturation-driven (5–12 overtones, always rich)
Rhythm:         Very slow (1.0s, 2.0s per note)
ADSR Envelope:  Attack=300ms, Decay=100ms, Sustain=80%, Release=400ms
                [Extremely long, smooth envelope for pads]
Level:          Quieter background layer (×0.4 gain reduction)
```

How it works:
1. Generates long, sustained notes from the image
2. 5–12 harmonics create a naturally dense, chorus-like texture
3. Very gentle ADSR ramping creates ambient, spacious feel
4. Mixed at lower volume so it doesn't dominate the composition


---

#### **🎼 Percussion Voice (Blue Channel Spectral Shaping)**

```
Purpose:        Rhythmic, textural element
Input Source:   Blue channel spectral energy (for timbre shaping)
Note Types:     Kick, Snare, Hi-Hat, or Silence
Hit Timing:     Edge density-driven
  • High edges → 40% silence, 60% drums → busy rhythm
  • Low edges → 60% silence, 40% drums → sparse hits
Hit Duration:   Random (0.125s, 0.25s)
```

**Kick Drum:**
- Spectral shape: Heavy low-frequency emphasis, rapid high-frequency rolloff
- Blue energy controls low-frequency emphasis
- Attack is quick, decay is long (body resonance)
- Frequency content: 50–150 Hz

**Snare Drum:**
- Spectral shape: Mid-frequency bump (around 2–5 kHz) with noise
- Creates the "snap" characteristic of snare drums
- Quick attack, medium decay
- Frequency content: 1–8 kHz

**Hi-Hat (Closed):**
- Spectral shape: High-frequency emphasis only (>4 kHz)
- Very short, percussive character
- Bright, metallic impact
- Frequency content: 4–16 kHz

All drums are generated using FFT-shaped noise bursts (not sine waves):
1. Generate white noise in frequency domain
2. Multiply by spectral shape (kick/snare/hihat envelope)
3. Apply random phase at each frequency bin
4. IFFT to time domain
5. Apply exponential decay envelope


---

### **Step 6: Mixing & Mastering**

After all four voices are generated, the final mix is processed:

#### **6a. Per-Voice Spectral Reverb**

```python
Reverb Amount (wet) = Saturation × 0.6  ∈ [0, 0.6]
Room Size = 0.2 + Entropy × 0.5  ∈ [0.2, 0.7] seconds
```

**How spectral reverb works:**
1. Generate random impulse response (exponentially decaying noise)
2. FFT both voice signal and impulse response
3. Multiply frequency spectra (convolution in frequency domain)
4. IFFT back to time domain
5. Blend with original via wet/dry mix

- **High saturation** → More reverb → Spacious, atmospheric
- **Low saturation** → Less reverb → Dry, intimate
- **High entropy** → Long reverb tails → Cathedral-like
- **Low entropy** → Short reverb → Small room feel

#### **6b. Voice Mixing**

```python
Voice         Level Weight
─────────────────────────────
Melody        1.0 (full)
Bass          0.7 (70%)
Pad           0.5 (50%)
Percussion    0.6 (60%)
```

Each voice is summed with these weights:
```
Final Mix = (Reverb(Melody) × 1.0) + 
            (Reverb(Bass) × 0.7) + 
            (Reverb(Pad) × 0.5) + 
            (Reverb(Percussion) × 0.6)
```

#### **6c. FFT Soft-Clipping (Harmonic Saturation)**

```python
Distortion Amount = Color Variance × 0.4  ∈ [0, 0.4]
```

**How FFT soft-clipping works:**
1. FFT the mixed signal
2. Normalize frequency magnitudes
3. Apply tanh() soft-clipping function: `clipped = tanh(x × (1 + distortion × 3))`
4. tanh() smoothly compresses peaks without harsh edges
5. Restore original phase information
6. IFFT back to time domain

**Result:**
- **Low color variance** → Clean, pristine sound
- **High color variance** → Warm, gently saturated, harmonically rich


#### **6d. Final Normalization & Clipping**

```
1. Find peak amplitude across entire mix
2. Scale to maximum ±1.0
3. Hard-clip to [-1.0, 1.0] to prevent numeric overflow
4. Convert to 16-bit integer (44,100 Hz sample rate)
5. Write to WAV file
```

---

## 📊 Parameter Summary Table

| Parameter | Image Source | Range | Musical Impact |
|---|---|---|---|
| Scale | Hue | 6 scales | Mood, tonality |
| Root Frequency | Brightness | C2–C6 (65–1046 Hz) | Octave register |
| Melody Harmonics | Contrast | 1–9 | Melody richness |
| Bass Harmonics | Red FFT Energy | 1–7 | Bass character |
| Pad Harmonics | Saturation | 5–12 | Pad density |
| Harmonic Rolloff | Color Variance | 0.3–0.9 | Tone warmth |
| Reverb Wet Mix | Saturation | 0–0.6 | Spatial depth |
| Reverb Room Size | Entropy | 0.2–0.7 s | Reverb tail length |
| Melody Rhythm | Edge Density | Varies | Note duration distribution |
| Percussion Density | Edge Density | Varies | Hit frequency |
| FFT Distortion | Color Variance | 0–0.4 | Harmonic saturation |

---

## 🛠️ Technical Stack

```
Language:    Python 3.8+
Audio:       scipy.io.wavfile (WAV writing), NumPy (audio processing)
Images:      PIL/Pillow (image loading and resizing)
DSP:         scipy.fft (FFT/IFFT), NumPy (signal processing)
Web UI:      Streamlit (interactive demo)
Visualization: Matplotlib (waveform/spectrogram display)
Animation:   matplotlib.animation (video export)
```

---

## 📦 Installation

### **Local Setup**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Image-to-Music-
   cd Image-to-Music-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### **Streamlit Web App (Recommended)**

Run the interactive Streamlit app locally:
```bash
streamlit run st_app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🎯 Usage

### **Via Streamlit (Easiest)**

1. Upload an image (PNG, JPG, or JPEG)
2. Adjust duration slider (5–30 seconds)
3. Click **"🎶 Generate Music!"**
4. Listen to the generated composition
5. View image analysis metrics
6. Explore individual voice levels

### **Via Command Line**

```bash
python musicgen2.py <image_path> <output.wav> [duration_seconds]
```

Example:
```bash
python musicgen2.py my_photo.jpg my_music.wav 15
```

---

## 📁 Project Structure

```
Image-to-Music-/
├── musicgen2.py           # Core synthesis engine (FFT-based)
├── st_app.py              # Streamlit web interface
├── visualizer.py          # Static spectrogram generation
├── requirements.txt       # Python dependencies
├── README.md              # This file
```

---

## 🎵 Example Workflows

### **Example 1: Sunset Photo**
- 🔴 **Hue**: Orange (0.15) → **Lydian scale** (dreamy, lifted)
- ☀️ **Brightness**: 0.7 → **C5 root** (bright register)
- 🌅 **Saturation**: 0.8 → **Heavy reverb** (spacious, atmospheric)
- ✨ **Edge Density**: 0.3 → **Sparse rhythm** (slow, contemplative)

**Result:** Dreamy, ambient composition with long sustained notes and warm harmonics.

---

### **Example 2: Forest Photo**
- 🟢 **Hue**: Green (0.45) → **Blues scale** (soulful, earthy)
- 🌲 **Brightness**: 0.4 → **C3 root** (darker register)
- 🍃 **Edge Density**: 0.7 → **Busy rhythm** (fast, energetic)
- 📊 **Contrast**: 0.6 → **Rich harmonics** (6–7 overtones)

**Result:** Soulful, energetic piece with quick note passages and earthy, warm tones.

---

### **Example 3: Abstract Noise**
- 🎨 **Mean Hue**: Undefined → Random scale
- ⚖️ **Color Variance**: 1.0 → **Maximum distortion** (saturated, warm)
- 🔀 **Edge Density**: 0.9 → **Very busy percussion** (dense rhythm)
- 🎛️ **Entropy**: High → **Long reverb tails** (spacious, experimental)

**Result:** Abstract, experimental soundscape with dense percussion and heavily processed tones.

---

## 🔬 The Science Behind It

### **Why FFT-Based Synthesis?**

1. **Efficiency**: Single IFFT vs. N sine oscillator additions
2. **Precision**: Exact harmonic frequencies without aliasing
3. **Integration**: Reverb and effects naturally operate in frequency domain
4. **Spectral Shaping**: Direct control over overtone structure

### **Why Scale Selection from Hue?**

- Warm hues (red/orange) → Major/Lydian (uplifting)
- Cool hues (blue/purple) → Minor/Phrygian (introspective)
- Biological color–emotion association → Musical parallel

### **Why Brightness → Octave?**

- Dark images → Low register (bass, grounding)
- Bright images → High register (treble, ethereal)
- Natural association: brightness ↔ perceived frequency

### **Why Edge Density → Rhythm?**

- Visual complexity → Musical complexity
- Busy images → Rapid note onsets
- Sparse images → Long sustained notes

---

## 📊 Visualization & Output

The project includes a **static spectrogram visualizer** that displays the frequency content analysis of the generated audio.

### **What's Displayed**

A spectrogram shows:
- **X-axis**: Time (seconds)
- **Y-axis**: Frequency (Hz, up to Nyquist frequency ~22 kHz)
- **Color intensity**: Power/magnitude in dB (darker = quieter, brighter = louder)

The spectrogram uses:
- 1024-point FFT window (detailed frequency resolution)
- 512-point overlap between windows (smooth time transitions)
- "Magma" colormap (perceptually uniform, publication-ready)
- Dynamic range: -80 to 0 dB

### **Generate Spectrograms**

#### **Via Streamlit (Easiest)**

In the Streamlit app after generating music, expand **"📊 Frequency Analysis (Spectrogram)"** to view the frequency analysis of the mixed signal instantly.

#### **Via Python**

```python
from musicgen2 import generate_image_music
from visualizer import generate_spectrogram

# Generate music
mix, voices, sr = generate_image_music('image.jpg', 'output.wav', 10)

# Create spectrogram
spectrogram_path = generate_spectrogram(mix, sr, 'spectrogram.png')

# Output: spectrogram.png (shows frequency content over time)
```

### **Render Time**

Spectrogram generation is **instant** (< 1 second) compared to video rendering. Perfect for quick iteration and exploration.

---

## 📝 Notes & Limitations

- **Deterministic**: Same image always generates identical music (seeded with image data)
- **Sample Rate**: Fixed at 44,100 Hz (CD quality)
- **Bit Depth**: 16-bit integer samples (standard WAV format)
- **Polyphony**: 4 voices at optimal computational cost (can extend to 8+ with changes)
- **Image Resolution**: Internally resized to 64×64 (any input size accepted)

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- 🎹 Additional scales or microtonality
- 🎶 Adaptive groove and swing patterns
- 🎸 Physical modeling for instrument timbres
- 🎼 MIDI export functionality
- 🎨 Real-time spectrogram visualization

---

## 📜 License

This project is open source. Please include attribution if used in research or publication.

---

## 🙏 Acknowledgments

- FFT-based synthesis inspired by academic DSP research
- Color → emotion mapping from perceptual psychology literature
- Streamlit for the intuitive web interface

---

## ❓ FAQ

**Q: Can I use copyrighted images?**  
A: Yes! The generated music is entirely new and original. No copyright applies to the output.

**Q: How does randomness work if the same image always generates the same music?**  
A: The random seed is derived from the image's pixel data, ensuring consistent results while maintaining variation across different images.

**Q: What audio formats are supported?**  
A: WAV (44,100 Hz, 16-bit) is the primary output. MIDI export is not currently supported.

**Q: Can I adjust musical parameters manually?**  
A: Currently no—the mapping from image to music is automatic. This ensures a consistent, deterministic experience.

**Q: What's the maximum duration?**  
A: The default is 10 seconds (adjustable in Streamlit UI). Longer durations are possible but require proportional computation time.

---

## 📞 Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Happy composing! 🎶✨**
