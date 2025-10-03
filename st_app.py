import streamlit as st
import tempfile
import os
from musicgen2 import generate_image_music
from visualizer import save_multi_voice_visualizer

st.set_page_config(page_title="🎨→🎶 Image-to-Music Game", layout="centered")
st.title("🎨 Image-to-Music Generator Game 🎶")
st.write("Upload an image, choose a scale, and see what music it creates!")

# Sidebar
scale_name = st.sidebar.selectbox("Scale", ["C_major", "A_minor"])
total_duration = st.sidebar.slider("Duration (seconds)", 5, 20, 10)
add_visualizer = st.sidebar.checkbox("Generate Visualizer Video", value=False)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Your Uploaded Image", use_column_width=True)

    if st.button("🎶 Generate Music!"):
        with st.spinner("Composing..."):
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_img.write(uploaded_file.getvalue())
            temp_img.close()

            output_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            vis_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name if add_visualizer else None

            # Generate music + optional visualizer
            mix, voices, sr, saved_wav, saved_vis = generate_image_music(
                temp_img.name,
                output_wav,
                scale_name=scale_name,
                total_duration=total_duration,
                save_visualizer=add_visualizer,
                visualizer_func=save_multi_voice_visualizer,
                vis_output=vis_file
            )

            st.success("Music generated!")
            st.audio(saved_wav, format="audio/wav")

            if saved_vis:
                st.video(saved_vis)

            # Clean up temp image
            os.remove(temp_img.name)
