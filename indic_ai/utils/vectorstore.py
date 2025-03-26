import os
import fitz
import aiofiles
import chromadb
import logging
import uuid
import requests
from fastapi import File
from datetime import datetime, timezone
from utils.embedder import EmbeddingHandler
from config.env import NVIDIA_NIM_API_KEY, NVIDIA_RERANK, NVIDIA_RERANK_URL
from utils.logger_config import logger

class VectorStoreHandler:
    def __init__(self, db_path="./chroma_db", collection_name="temples"):
        logger.info("Initializing VectorStoreHandler...")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        self.embedder = EmbeddingHandler()
        logger.info("VectorStoreHandler initialized successfully.")
    
    def add_text(self, case_id: str, text: str):
        logger.info(f"Adding text for case_id: {case_id}")
        embeddings, text_chunks = self.embedder.get_document_embeddings(512, 50, text)
        existing_results = self.collection.get(where={"case_id": case_id})
        existing_chunks = set(existing_results.get("documents", []))
        
        for chunk, embedding in zip(text_chunks, embeddings):
            if chunk in existing_chunks:
                continue  # Skip duplicate chunks
            doc_id = str(uuid.uuid4())
            self.collection.add(
                ids=[doc_id], embeddings=[embedding], documents=[chunk], metadatas=[{"case_id": case_id}]
            )
        logger.info("Text added successfully.")
    
    async def add_pdf(self, case_id: str, file: File):
        logger.info(f"Processing PDF for case_id: {case_id}")
        pdf_bytes = await file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join([page.get_text("text") for page in pdf_document])
        self.add_text(case_id, text)
        logger.info("PDF processed and stored successfully.")
    
    def rerank_documents(self, query, documents):
        if not documents:
            return []
        logger.info("Reranking retrieved documents...")
        headers = {"Authorization": f"Bearer {NVIDIA_NIM_API_KEY}", "Accept": "application/json", "Content-Type": "application/json"}
        payload = {"model": NVIDIA_RERANK.strip(), "query": {"text": query}, "passages": [{"text": doc} for doc in documents]}
        
        try:
            response = requests.post(NVIDIA_RERANK_URL, headers=headers, json=payload)
            response.raise_for_status()
            reranked_results = response.json().get("reranked_passages", [])
            return [doc["text"] for doc in reranked_results] if reranked_results else documents
        except requests.exceptions.RequestException as e:
            logger.error(f"Reranker API Error: {e}")
            return documents
    
    def get_documents(self, case_id: str):
        logger.info(f"Fetching documents for case_id: {case_id}")
        try:
            case_documents = self.collection.get(where={"case_id": case_id})
            documents = case_documents.get("documents", [])
            logger.info(f"Retrieved {len(documents)} documents.")
            return documents if documents else {"message": f"No documents found for case_id: {case_id}"}
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return {"error": str(e)}
    
    def search_documents(self, case_id: str, query: str, top_k: int = 3):
        logger.info(f"Searching vector store with case_id: {case_id}, query: {query}")
        query_embedding = self.embedder.get_embedding(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, where={"case_id": case_id})
        retrieved_docs = results.get("documents", [[]])[0] if "documents" in results else []
        reranked_docs = self.rerank_documents(query, retrieved_docs) if retrieved_docs else []
        logger.info(f"Retrieved {len(reranked_docs)} relevant documents.")
        return reranked_docs