import fitz
import chromadb
import uuid
import requests
from fastapi import File
from utils.embedder import EmbeddingHandler
from config.env import NVIDIA_NIM_API_KEY, NVIDIA_RERANK, NVIDIA_RERANK_URL
from utils.logger_config import logger
import numpy as np


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


class StoreUtils:
    def __init__(self, vector_store_handler):
        logger.info("Initializing StoreUtils...")
        self.collection = vector_store_handler.collection
        self.complete_collection = vector_store_handler.complete_collection
        self.embedder = EmbeddingHandler()
        logger.info("StoreUtils initialized successfully.")

    def get_temple_names(self, complete):
        """Fetch all unique case_ids stored in the collection's metadata."""
        try:
            # Retrieve only metadata from the collection
            results = (self.complete_collection if complete else self.collection).get(
                include=["metadatas"]
            ) or {}

            metadata_list = results.get("metadatas", [])

            # Extract unique case_ids from metadata
            unique_case_ids = set()
            for meta in metadata_list:
                if meta and "case_id" in meta:
                    unique_case_ids.add(meta["case_id"])

            logger.info(f"Retrieved {len(unique_case_ids)} unique case_ids")
            return list(unique_case_ids)

        except Exception as e:
            logger.error(f"Error fetching unique case_ids: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def get_existing_temple_names(self, complete: bool = False):
        """Retrieve existing temple names from the ChromaDB vector database."""
        try:
            existing_entries = (
                self.complete_collection.get() if complete else self.collection.get()
            )

            # ChromaDB returns a dict with a key "documents"
            if not existing_entries or "documents" not in existing_entries:
                logger.warning("No documents found in the database.")
                return set()

            # Extract existing temple names from the documents list
            existing_names = set(existing_entries["documents"])
            logger.info(f"Retrieved {len(existing_names)} existing temple names.")

            return existing_names

        except Exception as e:
            logger.error(f"Error fetching existing temple names: {e}", exc_info=True)
            return set()

    def add_temple_embeddings(self, complete: bool = False):
        try:
            unique_case_ids = self.get_temple_names(complete)

            if not unique_case_ids:
                logger.info("No unique case IDs found.")
                return {"message": "No unique case IDs found."}
            # Fetch existing temple names from the vector database
            existing_temple_names = self.get_existing_temple_names(complete)

            for case_id in unique_case_ids:
                if case_id in existing_temple_names:
                    logger.info(
                        f"Skipping embedding for {case_id} as it already exists in DB."
                    )
                    continue
                embedding = self.embedder.get_embedding(case_id)
                if not embedding:
                    logger.warning(
                        f"Failed to generate embedding for case id: {case_id}"
                    )
                    continue
                # Create a unique document ID.
                doc_id = str(uuid.uuid4())

                # Add the test embedding to the collection.
                if complete:
                    self.complete_collection.add(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[case_id],
                        metadatas=[{"tag": "test_embeds"}],
                    )
                    logger.info(
                        f"Added test embedding for case id: {case_id} into complete collection"
                    )
                else:
                    self.collection.add(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[case_id],
                        metadatas=[{"tag": "test_embeds"}],
                    )
                    logger.info(f"Added test embedding for case id: {case_id}")

            return {"message": "Test embeddings added successfully."}

        except Exception as e:
            logger.error(f"Error adding test embeddings: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def get_temple_embeddings(self, complete: bool = False):
        if complete:
            case_documents = self.complete_collection.get(
                where={"tag": "test_embeds"}, include=["embeddings", "documents"]
            )
        else:
            case_documents = self.collection.get(
                where={"tag": "test_embeds"}, include=["embeddings", "documents"]
            )
            
        documents = case_documents.get("documents", [])
        embeddings = case_documents.get("embeddings", [])

        return embeddings, documents

    def get_best_match(self, query: str,complete:bool=False):
        """
        Given a query text, compute its embedding and compare it to the stored test embeddings
        (with metadata tag "test_embeds") for each unique case ID.
        Returns the case ID with the highest semantic similarity.
        """
        try:
            # Compute embedding for the query text.
            query_embedding = self.embedder.get_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding.")
                return {"error": "Failed to generate query embedding."}

            stored_embeddings, stored_caseids = self.get_temple_embeddings(complete)

            best_similarity = -1.0
            best_case_id = None

            # Compare query embedding with each stored test embedding.
            for emb, meta in zip(stored_embeddings, stored_caseids):
                similarity = cosine_similarity(query_embedding, emb)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_case_id = meta

            if best_case_id is None:
                logger.info("No matching case id found.")
                return {"message": "No matching case id found."}

            logger.info(
                f"Best matching case id: {best_case_id} with similarity: {best_similarity:.4f}"
            )
            return {"best_case_id": best_case_id, "similarity": best_similarity}

        except Exception as e:
            logger.error(f"Error in semantic comparison: {str(e)}", exc_info=True)
            return {"error": str(e)}


class VectorStoreHandler:
    def __init__(self, db_path="./chroma_db", collection_name="temples"):
        logger.info("Initializing VectorStoreHandler...")
        self.chroma_client = chromadb.PersistentClient(path=db_path)

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        self.complete_collection = self.chroma_client.get_or_create_collection(
            name="complete_info"
        )
        self.embedder = EmbeddingHandler()
        self.store_utils = StoreUtils(self)
        logger.info("VectorStoreHandler initialized successfully.")

    def add_complete_text(self, case_id: str, text: str):
        logger.info(f"Adding complete information for temple: {case_id}")

        embeddings, text_chunks = self.embedder.get_document_embeddings(512, 50, text)

        existing_results = self.complete_collection.get(where={"case_id": case_id})
        existing_chunks = set(existing_results.get("documents", []))

        for chunk, embedding in zip(text_chunks, embeddings):
            if chunk in existing_chunks:
                continue  # Skip duplicate chunks
            doc_id = str(uuid.uuid4())
            self.complete_collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"case_id": case_id}],
            )
        logger.info("Complete information is added successfully.")

    def add_text(self, case_id: str, text: str, chunk_overlap: int = 50):
        logger.info(f"Adding text for case_id: {case_id}")
        embeddings, text_chunks = self.embedder.get_document_embeddings(512, chunk_overlap, text)
        existing_results = self.collection.get(where={"case_id": case_id})
        existing_chunks = set(existing_results.get("documents", []))

        for chunk, embedding in zip(text_chunks, embeddings):
            if chunk in existing_chunks:
                continue  # Skip duplicate chunks
            doc_id = str(uuid.uuid4())
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"case_id": case_id}],
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
        headers = {
            "Authorization": f"Bearer {NVIDIA_NIM_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "model": NVIDIA_RERANK.strip(),
            "query": {"text": query},
            "passages": [{"text": doc} for doc in documents],
        }

        try:
            response = requests.post(NVIDIA_RERANK_URL, headers=headers, json=payload)
            response.raise_for_status()
            reranked_results = response.json().get("reranked_passages", [])
            return (
                [doc["text"] for doc in reranked_results]
                if reranked_results
                else documents
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Reranker API Error: {e}")
            return documents

    def get_documents(self, case_id: str):
        logger.info(f"Fetching documents for case_id: {case_id}")
        try:
            best_match = self.store_utils.get_best_match(case_id,complete=True).get("best_case_id")
            print(f"[DEBUG] best_match: {best_match}")
            case_documents = self.collection.get(where={"case_id": best_match})
            documents = case_documents.get("documents", [])
            logger.info(f"Retrieved {len(documents)} documents.")
            return (
                documents
                if documents
                else {"message": f"No documents found for case_id: {case_id}"}
            )
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return {"error": str(e)}

    def get_tts_documents(self,case_id:str):
        logger.info(f"Fetching documents for case_id: {case_id}")
        try:
            best_match = self.store_utils.get_best_match(case_id).get("best_case_id")
            print(f"[DEBUG] best_match: {best_match}")
            case_documents = self.collection.get(where={"case_id": best_match})
            documents = case_documents.get("documents", [])
            logger.info(f"Retrieved {len(documents)} documents.")
            return (
                documents
                if documents
                else {"message": f"No documents found for case_id: {case_id}"}
            )
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return {"error": str(e)}
                
    def search_temple_info(self, temple_id: str, query: str, top_k: int = 3):
        """This function is limited to specific temple alone"""
        logger.info(f"Searching vector store with case_id: {temple_id}, query: {query}")

        best_match = self.store_utils.get_best_match(temple_id).get("best_case_id")
        query_embedding = self.embedder.get_embedding(query)

        results = self.complete_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"case_id": best_match},
        )

        retrieved_docs = (
            results.get("documents", [[]])[0] if "documents" in results else []
        )
        reranked_docs = (
            self.rerank_documents(query, retrieved_docs) if retrieved_docs else []
        )

        logger.info(f"Retrieved {len(reranked_docs)} relevant documents.")
        return reranked_docs

    def search_documents(self, query: str, top_k: int = 3):
        """This is a general purpose Rag"""
        logger.info(f"Searching vector store for query: {query}")

        query_embedding = self.embedder.get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )

        retrieved_docs = (
            results.get("documents", [[]])[0] if "documents" in results else []
        )
        reranked_docs = (
            self.rerank_documents(query, retrieved_docs) if retrieved_docs else []
        )

        logger.info(f"Retrieved {len(reranked_docs)} relevant documents.")
        return reranked_docs
