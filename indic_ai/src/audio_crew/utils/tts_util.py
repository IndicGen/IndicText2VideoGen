import os,re,time
from dotenv import load_dotenv
load_dotenv()

from smallest import Smallest
from pydub import AudioSegment
from utils.logger_config import logger
from config.env import NVIDIA_NIM_API_KEY
import subprocess


class TTSProcessor:

    def __init__(self):
        self.smallest_ai_api = os.getenv("SMALLEST_API_KEY")
        self.api_call_count = 0
        self.last_api_call_time = 0
        self.smallest_ai_client = Smallest(api_key=self.smallest_ai_api)

        self.base_audio_folder = (
            "audio_output"
        )
        self.output_folder = os.path.join(self.base_audio_folder, "temp_output")
        self.final_output = os.path.join(self.base_audio_folder, "final_audios")

        # Ensure directories exist
        os.makedirs(self.base_audio_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.final_output, exist_ok=True)

    def clean_text(self, text):
        """Cleans and formats text for smooth TTS processing."""
        logger.info("Cleaning text for TTS processing.")
        text = re.sub(r"##\s*", "", text)
        text = re.sub(r"\*\*\s*", "", text)
        text = re.sub(r"\*\s*", "", text)
        text = re.sub(r"\n+", ". ", text)
        text = re.sub(r"(\d+)\.\s*", r"Point \1. ", text)

        if not text.endswith("."):
            text += "."

        logger.info("Text cleaned successfully.")
        return text

    def synthesize_audio_nvidia(
        self,
        text: str,
        temple_name: str,
        voice: str = "Magpie-Multilingual.EN-US.Male.Neutral",
        function_id: str = "877104f7-e885-42b9-8de8-f6e4c6303969",
        server: str = "grpc.nvcf.nvidia.com:443",
    ):

        output_file = os.path.join(self.output_folder, f"{temple_name}.wav")
        command = [
            "python",
            "E:/Coding/OpenSource/blog_post/git_repo/IndicText2VideoGen/indic_ai/python-clients/scripts/tts/talk.py",
            "--server",server,
            "--use-ssl",
            "--metadata",
            "function-id",function_id,
            "--metadata",
            "authorization",f"Bearer {NVIDIA_NIM_API_KEY}",
            "--text",text,
            "--voice",voice,
            "--output",output_file,
        ]
        try:
            subprocess.run(command, check=True)
            print(f"Audio generated and saved to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating audio: {e}")
        pass

    def synthesize_audio_smallestai(
        self, text, temple_name, voice="raman", speed=1.0, sample_rate=24000
    ):
        """Synthesizes speech from text and saves it as an audio file."""
        if not text.strip():
            raise ValueError("Text cannot be empty for TTS synthesis.")

        try:
            output_file = os.path.join(self.output_folder, f"{temple_name}.wav")
            self.smallest_ai_client.synthesize(
                text,
                save_as=output_file,
                voice_id=voice,
                speed=speed,
                sample_rate=sample_rate,
            )

            self.api_call_count += 1
            if self.api_call_count % 4 == 0:
                print(
                    "Rate limit: Waiting for 60 seconds before the next batch of API calls..."
                )
                time.sleep(60)

            return output_file
        except Exception as e:
            if "Rate Limited" in str(e):
                raise ValueError("Rate limited by TTS API. Please wait and retry.")
            raise ValueError(f"TTS Synthesis failed: {e}")

    def get_sorted_audio_files(self, folder_path, prefix, extension=".wav"):
        """Fetches and sorts all audio files matching the given prefix and extension numerically."""
        files = [
            f
            for f in os.listdir(folder_path)
            if f.startswith(prefix) and f.endswith(extension)
        ]
        files.sort(key=lambda f: int(re.search(r"_(\d+)", f).group(1)))
        return [os.path.join(folder_path, f) for f in files]

    def stitch_audio_files(self, temple_name, output_file="stitched_audio.wav"):
        """Merges multiple audio files in a folder into a single audio file."""
        audio_files = self.get_sorted_audio_files(
            self.output_folder, prefix=temple_name
        )

        if not audio_files:
            raise ValueError("No matching audio files found for stitching.")

        logger.info(f"Stitching {len(audio_files)} audio files...")
        combined_audio = AudioSegment.from_file(audio_files[0])
        for file in audio_files[1:]:
            next_audio = AudioSegment.from_file(file)
            combined_audio += next_audio

        output_path = os.path.join(self.final_output, f"{temple_name}_{output_file}")
        combined_audio.export(output_path, format="wav")
        logger.info(f"Stitched audio saved at: {output_path}")

        logger.info(f"Deleting the temporary audio files for {temple_name}.")
        for file in audio_files:
            os.remove(file)
            logger.info(f"Deleted: {file}")
        logger.info(f"Temporary files are deleted for {temple_name}")
        return output_path
