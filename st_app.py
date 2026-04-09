import streamlit as st
import tempfile
import os
from musicgen2 import generate_image_music, analyze_image, choose_scale
from visualizer import save_multi_voice_visualizer

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

        # --- Multi-Voice Visualization ---
        with st.expander("📹 Animated Visualization (Waveforms + Spectrogram)", expanded=False):
            st.info(
                "🎨➜🎶 Rendering visualization with all 4 voices (Melody, Bass, Pad, Percussion) "
                "and spectrogram analysis. This may take 10–30 seconds..."
            )
            
            viz_placeholder = st.empty()
            viz_status = st.empty()
            
            try:
                viz_status.info("⏳ Generating visualization video...")
                output_video = os.path.join(tempfile.gettempdir(), "image_to_music_viz.mp4")
                
                save_multi_voice_visualizer(
                    voices, 
                    sr, 
                    temp_img.name, 
                    output_video, 
                    fps=30
                )
                
                # Display video in Streamlit
                with open(output_video, "rb") as video_file:
                    video_bytes = video_file.read()
                
                viz_status.success("✓ Visualization complete!")
                st.video(video_bytes)
                
                st.markdown("""
                **Visualization Layers:**
                - 🎼 **Melody (Green)**: Lead melodic line from green channel brightness
                - 🎼 **Bass (Red)**: Deep foundation from red channel, one octave lower
                - 🎼 **Pad (Purple)**: Lush atmospheric layer with rich harmonics
                - 🥁 **Percussion (Blue)**: Kick, snare, and hi-hat drums
                - 📊 **Spectrogram**: Full mix frequency analysis (FFT magma colormap)
                """)
                
                # Clean up video file
                if os.path.exists(output_video):
                    os.remove(output_video)
                    
            except Exception as e:
                viz_status.error(f"⚠️ Visualization rendering failed: {str(e)}")
                st.write("You can still listen to the audio above. Visualization requires ffmpeg.")

        # Clean up
        os.remove(temp_img.name)