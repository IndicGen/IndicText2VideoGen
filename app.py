import os
import json
import uuid
import time
import requests
from smallest import Smallest
import streamlit as st
import openai
import fitz  # PyMuPDF
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip, VideoFileClip
import tempfile

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    print("Extracted text from PDF:", text[:500])  # Debug print for first 500 characters
    return text

def extract_images_from_pdf(pdf_file):
    """Extract images from PDF and return their paths."""
    pdf_document = fitz.open(pdf_file)
    image_paths = []
    
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            image_filename = f"pdf_image_{page_number+1}_{img_index+1}.{image_ext}"
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)
            image_paths.append(image_filename)
    
    pdf_document.close()
    return image_paths

def synthesize_tts(api_key, text, voice="radhika", speed=1.0, sample_rate=24000):
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

def generate_full_script(api_key, text):
    """Generate a full script without labels, titles, or extraneous markers."""
    openai.api_key = api_key

    # prompt = (
    #     "Write a detailed, professional script for a YouTube video about a temple. "
    #     "The script should be comprehensive and divided into these sections logically but "
    #     "WITHOUT explicit section titles or labels: "
    #     "Temple Name, Location and Main Deity; Historical Background; Architecture Details and Idols in the temple; "
    #     "Unique Cultural Features and environment surrounding the temple; and How to get to the temple. "
    #     f"Here is the text to base the script on: \"{text}\""
    # )

    prompt = (
        "Write a detailed, professional script for a YouTube video about a Hindu temple. "
        "For any Indian or Hindu terms (names, places, objects, rituals), please: \n"
        "1. Add hyphens between syllables\n"
        "2. Double the vowels where they should be elongated\n"
        "3. Add pronunciation hints in parentheses for complex terms\n"
        "Example: 'Shi-vaa' instead of 'Shiva', 'Krish-naa' instead of 'Krishna'\n"
        "The script should be comprehensive and divided into these sections logically but "
        "WITHOUT explicit section titles or labels: "
        "Temple Name, Location and Main Deity; Historical Background; Architecture Details and Idols in the temple; "
        "Unique Cultural Features and environment surrounding the temple; and How to get to the temple. "
        f"Here is the text to base the script on: \"{text}\""
    )

    try:
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": "You are an expert scriptwriter for YouTube videos."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     max_tokens=1500,
        #     temperature=0.7
        # )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert scriptwriter for YouTube videos about Hindu temples. "
                              "Ensure all Indian names and terms are written with correct pronunciation guidance."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )

        if "choices" in response and len(response["choices"]) > 0:
            full_script = response["choices"][0]["message"]["content"].strip()
            print("Generated full script:", full_script[:500])  # Debug print for first 500 characters
            return full_script
        else:
            raise ValueError("Invalid response from OpenAI.")
    except Exception as e:
        raise ValueError(f"Error generating full script: {e}")

def split_script_into_sections(full_script):
    """Split the full script into five sections based on paragraph position."""
    sections = {
        "Temple Name, Location and Main Deity": "",
        "Historical Background": "",
        "Architecture Details and Idols in the temple": "",
        "Unique Cultural Features and environment surrounding the temple": "",
        "How to get to the temple": ""
    }
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in full_script.split('\n\n') if p.strip()]
    
    if not paragraphs:
        return sections
        
    # Calculate sections
    section_size = max(1, len(paragraphs) // 5)
    section_titles = list(sections.keys())
    
    for i, title in enumerate(section_titles):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < 4 else None
        section_content = ' '.join(paragraphs[start_idx:end_idx])
        sections[title] = section_content if section_content else f"Content not available for {title}."
    
    print("Split sections:", sections)  # Debug print
    return sections

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

        
def create_pdf_images_videos(pdf_images):
    """Create two videos from PDF images, handling any number of images."""
    if not pdf_images:
        return None, None
    
    # Handle different image count scenarios
    total_images = len(pdf_images)
    if total_images <= 5:
        # If 5 or fewer images, create only first video
        first_images = pdf_images
        remaining_images = []
    else:
        # If more than 5 images, split into two groups
        first_images = pdf_images[:5]
        remaining_images = pdf_images[5:]
    
    # Create first video if there are any images
    first_video = None
    if first_images:
        clips = [ImageClip(img).set_duration(2) for img in first_images]
        first_video = concatenate_videoclips(clips, method="compose")
        first_video.write_videofile("pdf_images_first.mp4", fps=24, codec="libx264")
        first_video = "pdf_images_first.mp4"
    
    # Create second video only if there are remaining images
    second_video = None
    if remaining_images:
        clips = [ImageClip(img).set_duration(2) for img in remaining_images]
        second_video = concatenate_videoclips(clips, method="compose")
        second_video.write_videofile("pdf_images_second.mp4", fps=24, codec="libx264")
        second_video = "pdf_images_second.mp4"
    
    return first_video, second_video        


def create_video_with_audio(images, audios, background_music_path, pdf_images=None, output_file="final_video.mp4", music_volume=0.1):
    """Create final video with PDF images at start and end."""
    video_clips = []
    
    # Create PDF videos and add first one
    if pdf_images:
        first_pdf_video, second_pdf_video = create_pdf_images_videos(pdf_images)
        if first_pdf_video:
            video_clips.append(VideoFileClip(first_pdf_video))
    
    # Add main content
    for img, audio in zip(images, audios):
        audio_clip = AudioFileClip(audio)
        image_clip = ImageClip(img).set_duration(audio_clip.duration).set_audio(audio_clip)
        video_clips.append(image_clip)
    
    # Add second PDF video if exists
    if pdf_images and second_pdf_video:
        video_clips.append(VideoFileClip(second_pdf_video))

    final_video = concatenate_videoclips(video_clips, method="compose")
    
    # Add background music
    background_music = AudioFileClip(background_music_path)
    total_duration = final_video.duration
    
    looped_music = CompositeAudioClip([
        background_music.volumex(music_volume).set_start(i * background_music.duration)
        for i in range(int(total_duration // background_music.duration) + 1)
    ]).subclip(0, total_duration)

    final_video = final_video.set_audio(CompositeAudioClip([final_video.audio, looped_music]))
    final_video.write_videofile(output_file, fps=24, codec="libx264")
    
    # Cleanup temporary files
    if first_pdf_video and os.path.exists(first_pdf_video):
        os.remove(first_pdf_video)
    if second_pdf_video and os.path.exists(second_pdf_video):
        os.remove(second_pdf_video)
    
    return output_file

def main():
    st.title("YouTube Shorts Scripts, TTS, and Image Generator")

    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", key="openai_api_key")
    smallest_api_key = st.text_input("Enter your Smallest API Key", type="password", key="smallest_api_key")
    background_music_file = st.file_uploader("Upload Background Music (MP3 or WAV)", type=["mp3", "wav"], key="background_music")

    uploaded_file = st.file_uploader("Upload a PDF File", type="pdf", key="uploaded_file")

    if not openai_api_key or not smallest_api_key or not background_music_file:
        st.warning("Please enter all required API keys and upload background music to proceed.")
        return

    if uploaded_file is not None:
        st.success("PDF file uploaded successfully!")

        try:
            input_text = extract_text_from_pdf(uploaded_file)
            st.write("Extracted text from PDF:", input_text[:5000])  # Debug print for first 500 characters
            main_temple_text = input_text.lower().split('main temple')[1].strip() if 'main temple' in input_text.lower() else input_text.strip()

            st.subheader("Generated Sections for Main Temple")

             # Extract images from PDF
            pdf_images = extract_images_from_pdf(uploaded_file)
            if pdf_images:
                st.success(f"Extracted {len(pdf_images)} images from PDF")

            # Step 1: Generate full script
            full_script = generate_full_script(openai_api_key, main_temple_text)
            st.write("Generated Full Script:", full_script[:5000])  # Debug print for first 500 characters

            # Step 2: Split into sections
            sections_scripts = split_script_into_sections(full_script)
            st.write("Split Sections Scripts:", sections_scripts) # Debug print for sections

            # Ensure the 'images/' directory exists
            os.makedirs("images", exist_ok=True)

            images = []
            audios = []
            
            for i, (section, script) in enumerate(sections_scripts.items(), start=1):
            #for i, (section, script) in list(enumerate(sections_scripts.items(), start=1))[:1]:
                st.markdown(f"### {section}")
                st.text_area(f"Script {i}: {section}", script, height=100, key=f"script_{i}")

                try:
                    if not script.strip():
                        raise ValueError(f"The script for {section} is empty. Skipping TTS synthesis.")

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
                        mime="audio/wav",
                        key=f"download_audio_{i}"
                    )

                    st.download_button(
                        label=f"Download Image for {section}",
                        data=open(image_path, "rb"),
                        file_name=os.path.basename(image_path),
                        mime="image/png",
                        key=f"download_image_{i}"
                    )

                except Exception as e:
                    st.error(f"Error generating assets for {section}: {e}")

            if images and audios:
                st.subheader("Creating Final Video")
                try:
                    #video_path = create_video_with_audio(images, audios, background_music_file)
                     # Save background music to temporary file
                    bg_music_path = save_uploaded_file(background_music_file)
                    # Update video creation call
                    video_path = create_video_with_audio(images, audios, bg_music_path, pdf_images=pdf_images, output_file="final_video.mp4")
                    st.video(video_path)
                    st.success(f"Video created successfully!")
                    st.download_button(
                        label="Download Video",
                        data=open(video_path, "rb"),
                        file_name=os.path.basename(video_path),
                        mime="video/mp4",
                        key="download_video"
                    )
                except Exception as e:
                    st.error(f"Error creating video: {e}")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
