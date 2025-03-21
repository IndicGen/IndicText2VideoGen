import requests
from config.env import NVIDIA_EMBEDDING_ENDPOINT,NVIDIA_EMBEDDING_MODEL, headers
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

class EmbeddingHandler:
    def __init__(self):
        pass
    def get_embedding(self,text: str):
        """Fetches embeddings for the given text from NVIDIA NIM embedding model."""
        payload = {
                    "model":NVIDIA_EMBEDDING_MODEL,
                    "input": text
                    }
        
        response = requests.post(NVIDIA_EMBEDDING_ENDPOINT, headers=headers, json=payload)

        if response.status_code == 200:
            json_response=json.loads(response.text)

            return json_response.get("data")[0].get("embedding", [])
        else:
            raise Exception(f"Embedding API Error: {response.text}")
    

    def get_document_embeddings(self,chunk_size:int,chunk_overlap:int,text:str):
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  
            chunk_overlap=chunk_overlap
        )
        text_chunks = text_splitter.split_text(text)
        embeddings = []

        for chunk in text_chunks:
            embedding= self.get_embedding(chunk)
            if embedding:
                embeddings.append(embedding)
        
        return embeddings,text_chunks

