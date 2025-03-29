from utils.vectorstore import VectorStoreHandler
from config.env import NVIDIA_NIM_API_KEY, NVIDIA_LLM_ENDPOINT,RAG_MODEL
from utils.logger_config import logger
from openai import OpenAI

class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAGPipeline...")
        self.vectorstore = VectorStoreHandler()
        self.client= OpenAI(
            base_url= NVIDIA_LLM_ENDPOINT,
            api_key= NVIDIA_NIM_API_KEY
        )
        logger.info("RAGPipeline initialized successfully.")

    def retrieve_relevant_documents(self, query: str, case_id: str, top_k: int = 3):
        logger.info(f"Retrieving documents for case_id: {case_id} with query: {query}")
        retrieved_docs = self.vectorstore.search_documents(case_id, query, top_k)
        logger.info(f"Retrieved {len(retrieved_docs)} documents.")
        return retrieved_docs

    def generate_response(self, query: str, case_id: str, top_k: int = 3):
        logger.info(f"Generating response for query: '{query}' with case_id: '{case_id}' and top_k: {top_k}")

        documents = self.retrieve_relevant_documents(query, case_id, top_k)
        
        if not documents:
            logger.warning(f"No relevant documents found for case_id: {case_id} and query: '{query}'")
            return {"response": "No relevant documents found."}

        context = "\n".join(documents)
        logger.info(f"Retrieved {len(documents)} documents. Constructing prompt...")

        prompt = f"""
        Context:\n{context}\n
        User Query: {query}\n
        Answer:
        """
        
        logger.debug(f"Generated prompt:\n{prompt}")

        try:
            completion = self.client.chat.completions.create(
                model=RAG_MODEL.strip(),
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
                temperature=0.2,
                top_p=0.7,
                max_tokens=10000,
                stream=False
            )
            
            response = completion.choices[0].message.content if completion.choices else "Error in LLM response"
            logger.info("Response generated successfully.")
        
        except Exception as e:
            logger.error(f"Error during response generation: {e}", exc_info=True)
            response = "Error in LLM response"

        return response
