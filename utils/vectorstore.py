import os
import fitz
import aiofiles
import chromadb
from fastapi import File,UploadFile
from datetime import datetime,timezone
from utils.embedder import EmbeddingHandler
import uuid,requests
from config.env import NVIDIA_NIM_API_KEY,NVIDIA_RERANK,NVIDIA_RERANK_URL

class VectorStoreHandler:
# Initialize ChromaDB
    def __init__(self,db_path="./chroma_db", collection_name="temples"):
        
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        self.embedder=EmbeddingHandler()

    # This has to be modified to accomodate logging keeping in mind the temple name while tagging
    def add_text(self,case_id: str, text:str):
        # Chunking and getting the embeddings
        embeddings, text_chunks = self.embedder.get_document_embeddings(512,50,text)   

        for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            doc_id = str(uuid.uuid4())  # Unique ID for each chunk
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"case_id": case_id}] # Tagging with case_id
            )
    
    async def add_pdf(self,case_id: str, file: File):
        # Processing the pdf
        pdf_bytes = await file.read()  # Use await to read file asynchronously
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join([page.get_text("text") for page in pdf_document])
        
        # Adding the pdf 
        self.add_text(case_id,text)

    def rerank_documents(self, query, documents):
        if not documents:
            return []
        headers = {
            "Authorization": f"Bearer {NVIDIA_NIM_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        # Construct payload in the format expected by NIM Reranker
        payload = {
            "model": NVIDIA_RERANK.strip(),  # Ensure correct model is used
            "query": {"text": query},
            "passages": [{"text": doc} for doc in documents]  # Convert docs to expected format
        }
        try:
            response = requests.post(NVIDIA_RERANK_URL, headers=headers, json=payload)
            response.raise_for_status()  # Raise error for non-200 responses

            # Extract reranked documents from response
            reranked_results = response.json().get("reranked_passages", [])
            return [doc["text"] for doc in reranked_results] if reranked_results else documents

        except requests.exceptions.RequestException as e:
            print(f"Reranker API Error: {e}")
            return documents  # Fallback to original order 
    
    def get_documents(self,case_id:str):
        
        try:
            case_documents = self.collection.get(where={"case_id": case_id})

            # If no documents found
            if not case_documents or "documents" not in case_documents:
                return {"message": f"No documents found for case_id: {case_id}"}

            # Extracting document texts and metadata
            documents = case_documents.get("documents", [])
            return documents
        
        except Exception as e:
            return {"error": str(e)}

    def search_documents(self,case_id: str,query: str, top_k: int = 3):
        """Searches for the most relevant documents using ChromaDB."""
        
        print(f"[DEBUG] Searching vector store with case_id: {case_id}, query: {query}")

        query_embedding = self.embedder.get_embedding(query)
        
        # This has to change to transcripts collection
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, where={"case_id": case_id} )

        retrieved_docs= results["documents"][0] if results["documents"] else []

        reranked_docs=self.rerank_documents(query,retrieved_docs) if retrieved_docs else []
        retrieved_metadata = results["metadatas"][0] if "metadatas" in results else []
        return reranked_docs
