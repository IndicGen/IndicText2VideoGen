import subprocess

# Replace with your actual API key
API_KEY = "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"
TEXT = "The 8.Thillai Nataraja Temple, Chidambaram, is dedicated to Lord Shiva as Nataraja, the Lord of Dance, and represents the element of space (Aakasha Lingam). It is one of the most famous temples in South India and is known for its unique architecture, including a gold-roofed stage that houses the idol of Nataraja. The temple complex covers a large area and features several shrines, mandapams, and a large temple tank. The temple's design and architecture are believed to have been conceived by the sage Patanjali and embody the connection between the human body and the universe."
VOICE = "Magpie-Multilingual.EN-US.Female.Female-1"  # Change based on available voices
OUTPUT_FILE = "output_audio.wav"

# Construct the command
command = [
    "python", "python-clients/scripts/tts/talk.py",
    "--server", "grpc.nvcf.nvidia.com:443",
    "--use-ssl",
    "--metadata", "function-id", "877104f7-e885-42b9-8de8-f6e4c6303969",
    "--metadata", "authorization", f"Bearer {API_KEY}",
    "--text", TEXT,
    "--voice", VOICE,
    "--output", OUTPUT_FILE
]

# Run the command
try:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("TTS Conversion Successful. Output saved to:", OUTPUT_FILE)
except subprocess.CalledProcessError as e:
    print("Error in TTS conversion:", e.stderr)
