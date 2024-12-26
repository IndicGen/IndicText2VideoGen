import os
import json
import uuid
import time
import requests
from smallest import Smallest
import streamlit as st
import openai
import fitz  # PyMuPDF
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def synthesize_tts(api_key, text, voice="radhika", speed=0.8, sample_rate=24000):
    if not text.strip():
        raise ValueError("Text cannot be empty for TTS synthesis.")
    try:
        client = Smallest(api_key=api_key)
        output_file = f"audio_output/sync_synthesize_{voice}_{uuid.uuid4().hex}.wav"
        client.synthesize(
            text,
            save_as=output_file,
            voice=voice,
            speed=speed,
            sample_rate=sample_rate
        )
        return output_file
    except Exception as e:
        if "Rate Limited" in str(e):
            raise ValueError("Rate limited by TTS API. Please wait and retry.")
        raise ValueError(f"TTS Synthesis failed: {e}")

def generate_sections_scripts(api_key, text):
    """Generate brief scripts for predefined sections using OpenAI's ChatCompletion API."""
    openai.api_key = api_key

    sections = [
        "Opening",
        "Historical Background",
        "Architecture Details",
        "Unique Cultural Features"
        ]

    scripts = {}
    for section in sections:
        messages = [
            {"role": "system", "content": "You are an expert scriptwriter for YouTube videos."},
            {
                "role": "user",
                "content": (
                    f"Write a brief script (1-2 sentences) for the section '{section}' for the following text about the temple: "
                    f"\"{text}\". Keep it engaging and concise. Provide script only for one main temple in the text. Do not mention the names of the '{section}' in the script."
                )
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            if "choices" in response and len(response["choices"]) > 0:
                scripts[section] = response["choices"][0]["message"]["content"].strip()
            else:
                raise ValueError("Invalid response structure from OpenAI API.")
        except Exception as e:
            raise ValueError(f"OpenAI API error for section {section}: {e}")

    return scripts

def generate_image_for_text(api_key, text):
    """Generate an image using OpenAI's DALL-E API based on the provided text."""
    openai.api_key = api_key
    retries = 3
    short_prompt = text[:1000]  # Ensure the prompt length is within the limit
    for attempt in range(retries):
        try:
            response = openai.Image.create(
                prompt=short_prompt,
                n=1,
                size="512x512"
            )
            if "data" in response and len(response["data"]) > 0:
                return response["data"][0]["url"]
            else:
                raise ValueError("Invalid response from DALL-E API.")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                raise ValueError(f"Image generation failed after {retries} attempts: {e}")

def create_video_with_audio(images, audios, background_music_path, output_file="temple_video.mp4", music_volume=0.1):
    """Create a video by combining images and audio clips."""
    video_clips = []

    for img, audio in zip(images, audios):
        audio_clip = AudioFileClip(audio)
        image_clip = ImageClip(img).set_duration(audio_clip.duration).set_audio(audio_clip)
        video_clips.append(image_clip)

    final_video = concatenate_videoclips(video_clips, method="compose")

    # Add background music
    background_music = AudioFileClip(background_music_path)
    total_duration = final_video.duration

    # Loop background music to match video duration
    looped_music = CompositeAudioClip([background_music.volumex(music_volume).set_start(i * background_music.duration)
                                        for i in range(int(total_duration // background_music.duration) + 1)]).subclip(0, total_duration)
    final_video = final_video.set_audio(CompositeAudioClip([final_video.audio, looped_music]))


    final_video.write_videofile(output_file, fps=24, codec="libx264")
    return output_file

def main():
    st.title("YouTube Shorts Scripts, TTS, and Image Generator")

    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    smallest_api_key = st.text_input("Enter your Smallest API Key", type="password")
    background_music_file = st.file_uploader("Upload Background Music (MP3 or WAV)", type=["mp3", "wav"], key="background_music")

    uploaded_file = st.file_uploader("Upload a PDF File", type="pdf")

    if not openai_api_key or not smallest_api_key:
        st.warning("Please enter both API keys to proceed.")
        return

    if uploaded_file is not None:
        st.success("PDF file uploaded successfully!")

        try:
            input_text = extract_text_from_pdf(uploaded_file)
            first_temple_text = input_text.split("\n\n")[0]  # Extract only the main temple text

            st.subheader("Generated Sections for Main Temple")

            sections_scripts = generate_sections_scripts(openai_api_key, first_temple_text)

            # Ensure the 'images/' directory exists
            os.makedirs("images", exist_ok=True)

            images = []
            audios = []

            for i, (section, script) in enumerate(sections_scripts.items(), start=1):
                st.markdown(f"### {section}")
                st.text_area(f"Script {i}: {section}", script, height=100)

                try:
                    audio_path = synthesize_tts(smallest_api_key, script)
                    audios.append(audio_path)
                    st.audio(audio_path, format="audio/wav")
                    st.success(f"Audio for {section} generated and saved at: {audio_path}")

                    image_url = generate_image_for_text(openai_api_key, script)
                    image_path = f"images/{section.lower().replace(' ', '_')}.png"
                    with open(image_path, "wb") as img_file:
                        img_file.write(requests.get(image_url).content)
                    images.append(image_path)
                    st.image(image_url, caption=f"Image for {section}")

                    st.download_button(
                        label=f"Download Audio for {section}",
                        data=open(audio_path, "rb"),
                        file_name=os.path.basename(audio_path),
                        mime="audio/wav"
                    )

                    st.download_button(
                        label=f"Download Image for {section}",
                        data=open(image_path, "rb"),
                        file_name=os.path.basename(image_path),
                        mime="image/png"
                    )

                except Exception as e:
                    st.error(f"Error generating assets for {section}: {e}")

            if images and audios:
                st.subheader("Creating Final Video")
                try:
                    # Save the uploaded background music file locally
                    background_music_path = f"background_music_{uuid.uuid4().hex}.mp3"
                    with open(background_music_path, "wb") as bg_music_file:
                        bg_music_file.write(background_music_file.read())

                    # Create the video with audio and background music
                    video_path = create_video_with_audio(images, audios, background_music_path)
                    st.video(video_path)
                    st.success(f"Video created successfully at: {video_path}")
                    st.download_button(
                        label="Download Video",
                        data=open(video_path, "rb"),
                        file_name=os.path.basename(video_path),
                        mime="video/mp4",
                        key="download_video"
                    )
        
                    # Clean up the temporary background music file
                    os.remove(background_music_path)
                except Exception as e:
                    st.error(f"Error creating video: {e}")
                finally:
                    # Explicitly close the background music file in MoviePy
                    try:
                        if 'background_music_path' in locals():
                            audio_clip = AudioFileClip(background_music_path)
                            audio_clip.close()
                    except Exception as close_error:
                        st.warning(f"Could not close background music file: {close_error}")

                    # Clean up the temporary background music file
                    if os.path.exists(background_music_path):
                        try:
                            os.remove(background_music_path)
                        except Exception as cleanup_error:
                            st.warning(f"Could not clean up temporary file: {cleanup_error}")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
