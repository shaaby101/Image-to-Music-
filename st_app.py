import streamlit as st
import tempfile
import os
from musicgen2 import generate_image_music, analyze_image, choose_scale
from visualizer import generate_spectrogram

st.set_page_config(page_title="🎨→🎶 Image-to-Music Generator", layout="centered")
st.title("🎨 Image-to-Music Generator 🎶")
st.write("Upload an image and the music is composed entirely from its visual properties.")

# Sidebar — scale is now auto-detected, only duration is user-controlled
total_duration = st.sidebar.slider("Duration (seconds)", 5, 30, 10)
st.sidebar.info(
    "**Scale, root note, harmonics, reverb, and rhythm** are all derived "
    "automatically from your image's hue, brightness, contrast, saturation, "
    "edge density, and colour variance."
)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Your Uploaded Image", use_container_width=True)

    if st.button("🎶 Generate Music!"):
        with st.spinner("Analysing image and composing..."):

            # Write upload to a temp file so PIL can open it by path
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_img.write(uploaded_file.getvalue())
            temp_img.close()

            output_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

            # --- Run the generator ---
            mix, voices, sr = generate_image_music(
                temp_img.name,
                output_wav,
                total_duration=total_duration,
            )

        st.success("✅ Music generated!")

        # --- Show what the image decided ---
        params     = analyze_image(temp_img.name)
        scale_name = choose_scale(params["mean_hue"])

        with st.expander("🔍 Image Analysis — what your image chose", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Scale",        scale_name)
                st.metric("Brightness",   f"{params['brightness']:.2f}")
                st.metric("Contrast",     f"{params['contrast']:.2f}")
                st.metric("Saturation",   f"{params['saturation']:.2f}")
            with col2:
                st.metric("Edge Density", f"{params['edge_density']:.2f}")
                st.metric("Color Var",    f"{params['color_var']:.2f}")
                st.metric("Entropy",      f"{params['entropy']:.2f}")
                harmonics = max(1, int(params["contrast"] * 8) + 1)
                st.metric("Harmonics",    harmonics)

        # --- Audio playback ---
        st.audio(output_wav, format="audio/wav")

        # --- Voice waveform preview ---
        with st.expander("🎚️ Voice Levels"):
            import numpy as np
            for name, voice in voices.items():
                rms = float(np.sqrt(np.mean(voice**2)))
                st.progress(min(rms * 4, 1.0), text=f"{name}  (RMS: {rms:.3f})")

        # --- Spectrogram Visualization ---
        with st.expander("📊 Frequency Analysis (Spectrogram)", expanded=False):
            st.info(
                "This spectrogram shows the frequency content of the generated music over time. "
                "Brighter colors indicate higher power/energy at those frequencies."
            )
            
            try:
                # Generate spectrogram
                spectrogram_path = os.path.join(tempfile.gettempdir(), "spectrogram.png")
                generate_spectrogram(mix, sr, spectrogram_path)
                
                # Display spectrogram
                if os.path.exists(spectrogram_path):
                    st.image(spectrogram_path, use_container_width=True)
                    st.caption("📊 Mixed Signal Spectrogram (FFT: 1024-point window, 512-point overlap)")
                    
                    # Clean up
                    os.remove(spectrogram_path)
                    
            except Exception as e:
                st.warning(f"⚠️ Could not generate spectrogram: {str(e)}")

        # Clean up
        os.remove(temp_img.name)