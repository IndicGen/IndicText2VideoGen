# IndicVideoGen - YouTube Shorts Scripts Generator using ChatGPT, Dall E and Smallest.ai Text to Speech

This project generates YouTube Shorts scripts, corresponding TTS (Text-to-Speech) audio, and images for an Indic Video Generation project based on the Kshetradanam temple blog description in a PDF. Finally, it combines these elements into a video using a scritable video combining library. 

[Kshetdranam Blog](https://kalyangeetha.wordpress.com/) is a rich collection of blogs containing information about hundreds of temples. 

![Screenshot 2024-12-27 065853](https://github.com/user-attachments/assets/b6ce67e8-d814-4cd2-8982-d7d36f1b6b98)

## Sample Video 1

https://github.com/user-attachments/assets/6f2ddb46-857f-4c30-8a52-df233a3048ca

## Sample Video 2

https://github.com/user-attachments/assets/b635b944-30fc-4747-85b1-3b59016dd5a9

## System Block Diagram

![Architecture](https://github.com/user-attachments/assets/b63de8a7-b347-41e7-922b-30fc7ea7d0e4)

## Features

- **Extract Text**: Extracts text from a PDF file, focusing on the main temple description.
- **Generate Scripts**: Divides the content into five sections: Opening, Historical Background, Architecture Details, Unique Features, and Call to Action.
- **Generate Audio**: Converts each script section into audio using the Smallest.ai TTS model with the voice `raman`.
- **Generate Images**: Creates images for each script section using OpenAI's DALL-E API.
- **Create Video**: Combines images and audio into a video.
- **Downloadable Assets**: Allows downloading individual images, audio clips, and the final video.

## Requirements

- Python 3.7+
- Libraries: Install the required libraries using the following:

  ```bash
  pip install -r requirements.txt
  ```

### Required Libraries

- `streamlit`
- `openai`
- `moviepy`
- `pymupdf`
- `requests`
- `smallest`

## Setup

1. Clone this repository:

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - **OpenAI API Key**: Needed for generating scripts and images.
   - **Smallest.ai API Key**: Needed for TTS audio generation.

## Running the Application

1. Run the Streamlit app:

   ```bash
   streamlit run <script_name>.py
   ```

2. Upload a PDF containing temple descriptions.
3. Enter your API keys.
4. Generate scripts, audio, images, and the video.

## How It Works

1. **Text Extraction**:
   - Extracts the first temple description from the uploaded PDF.

2. **Script Generation**:
   - Uses OpenAI's ChatGPT model to create concise scripts for predefined sections.

3. **Audio Generation**:
   - Converts each script section into audio using Smallest.ai TTS.

4. **Image Generation**:
   - Uses OpenAI's DALL-E to generate a corresponding image for each script section.

5. **Video Creation**:
   - Combines the generated images and audio clips into a video using MoviePy.

## Output

- **Audio Files**: Individual audio files for each script section.
- **Images**: Individual images generated for each script section.
- **Video**: A final video combining images and audio for all sections.

## Notes

- Ensure that the `images/` directory exists for saving images.
- Each Streamlit widget is assigned a unique `key` to prevent re-running issues during asset downloads.

## Troubleshooting

1. **Rate Limits**: Ensure you do not exceed API limits for OpenAI or Smallest.ai.
2. **Missing Directories**: If errors occur during file saving, ensure the required directories (`images/`) exist.
3. **Streamlit Re-runs**: Widget keys prevent re-runs during downloads. Ensure each widget key is unique.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

