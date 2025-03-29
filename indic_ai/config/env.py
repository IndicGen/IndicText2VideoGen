import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Fetch variables
NVIDIA_NIM_API_KEY = os.getenv("NVIDIA_NIM_API_KEY")
NVIDIA_LLM_ENDPOINT = os.getenv("NVIDIA_LLM_ENDPOINT")

# LLMs
LLM_MODEL = os.getenv("LLM_MODEL")
RAG_MODEL = os.getenv("RAG_MODEL")
NVIDIA_GUARD = os.getenv("NVIDIA_GUARD")

# Reranking models for RAG
NVIDIA_RERANK = os.getenv("NVIDIA_RERANK")
NVIDIA_RERANK_URL = os.getenv("NVIDIA_RERANK_URL")

# Embedding models for RAG
NVIDIA_EMBEDDING_ENDPOINT = os.getenv("NVIDIA_EMBEDDING_ENDPOINT")
NVIDIA_EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL")

# Debugging
if not NVIDIA_NIM_API_KEY:
    raise ValueError(" is not loaded. Check your .env file!")

headers = {
    "Authorization": f"Bearer {NVIDIA_NIM_API_KEY}",
    "Content-Type": "application/json",
}
