import requests
import json
from config.env import NVIDIA_EMBEDDING_ENDPOINT, NVIDIA_EMBEDDING_MODEL, headers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger_config import logger

class EmbeddingHandler:
    def __init__(self):
        logger.info("Initializing EmbeddingHandler...")

    def get_embedding(self, text: str):
        """Fetches embeddings for the given text from NVIDIA NIM embedding model."""
        logger.info("Fetching embedding for text of length %d", len(text))
        payload = {"model": NVIDIA_EMBEDDING_MODEL, "input": text}
        
        try:
            response = requests.post(NVIDIA_EMBEDDING_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            json_response = response.json()
            embedding = json_response.get("data", [{}])[0].get("embedding", [])
            logger.info("Successfully retrieved embedding of length %d", len(embedding))
            return embedding
        except requests.exceptions.RequestException as e:
            logger.error("Embedding API Request Error: %s", str(e))
            return []
        except Exception as e:
            logger.error("Unexpected error fetching embedding: %s", str(e))
            return []
    
    def get_document_embeddings(self, chunk_size: int, chunk_overlap: int, text: str):
        """Splits text into chunks and retrieves embeddings for each chunk."""
        logger.info("Splitting text into chunks with size %d and overlap %d", chunk_size, chunk_overlap)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_chunks = text_splitter.split_text(text)
        
        logger.info("Generated %d text chunks", len(text_chunks))
        embeddings = []
        
        for i, chunk in enumerate(text_chunks):
            embedding = self.get_embedding(chunk)
            if embedding:
                embeddings.append(embedding)
                logger.debug("Successfully retrieved embedding for chunk %d", i + 1)
        
        logger.info("Completed generating embeddings for document.")
        return embeddings, text_chunks