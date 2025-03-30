from utils.vectorstore import VectorStoreHandler
from config.env import NVIDIA_NIM_API_KEY, NVIDIA_LLM_ENDPOINT,RAG_MODEL, TAVILY_API_KEY
from utils.logger_config import logger
from openai import OpenAI
import requests


class GeneralRag:
    def __init__(self):
        logger.info("Initializing RAGPipeline...")
        self.vectorstore = VectorStoreHandler()
        self.client= OpenAI(
            base_url= NVIDIA_LLM_ENDPOINT,
            api_key= NVIDIA_NIM_API_KEY
        )
        logger.info("RAGPipeline initialized successfully.")

    def retrieve_relevant_documents(self, query: str, top_k: int = 3):
        logger.info(f"Retrieving documents for query: {query}")
        retrieved_docs = self.vectorstore.search_documents(query, top_k)
        logger.info(f"Retrieved {len(retrieved_docs)} documents.")
        return retrieved_docs
    
    def search_web(self, query: str):
        """Perform a web search using Tavily API."""
        try:
            logger.info(f"Performing web search for query: '{query}' using Tavily API.")
            
            headers = {"Content-Type": "application/json"}
            payload = {
                "api_key": TAVILY_API_KEY, 
                "query": query,
                "search_depth": "basic",
                "num_results": 3  
            }
            
            response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
            response.raise_for_status()
            results = response.json()

            if "results" in results:
                web_results = [result["content"] for result in results["results"][:3]]
                logger.info(f"Retrieved {len(web_results)} web search results.")
                return web_results
            else:
                logger.warning("No web search results found.")
                return []

        except Exception as e:
            logger.error(f"Web search error: {e}", exc_info=True)
            return []
    
    def generate_response(self, query: str, top_k: int = 3):
        logger.info(f"Generating response for query: '{query}'and top_k: {top_k}")

        documents = self.retrieve_relevant_documents(query, top_k)
        
        if not documents:
            logger.info(f"No relevant documents found locally for '{query}', using web search.")
            web_results = self.search_web(query)
            logger.info(f"Webs search results: {web_results}")
            if not web_results:
                return {"response": "No relevant information found locally or online."}
            context = "\n".join(web_results)
        else:
            # If documents are found, still use web search for additional context
            combined_results = documents
            context = "\n".join(combined_results)

        logger.info(f"Using {len(context.splitlines())} lines of context for response generation.")

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

class TempleRag:
    """The dataset that this should refer to should have all the information present in the blog about this temple."""
    def __init__(self):
        logger.info("Initializing TempleRag...")
        self.vectorstore = VectorStoreHandler()
        self.client= OpenAI(
            base_url= NVIDIA_LLM_ENDPOINT,
            api_key= NVIDIA_NIM_API_KEY
        )
        logger.info("TempleRag initialized successfully.")

    def retrieve_relevant_documents(self, query: str, temple_id: str, top_k: int = 3):
        logger.info(f"Retrieving documents for temple: {temple_id} with query: {query}")
        retrieved_docs = self.vectorstore.search_temple_info(temple_id, query, top_k)
        logger.info(f"Retrieved {len(retrieved_docs)} documents.")
        return retrieved_docs
    
    def search_web(self, query: str):
        """Perform a web search using Tavily API."""
        try:
            logger.info(f"Performing web search for query: '{query}' using Tavily API.")
            
            headers = {"Content-Type": "application/json"}
            payload = {
                "api_key": TAVILY_API_KEY, 
                "query": query,
                "search_depth": "basic",
                "num_results": 3  
            }
            
            response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
            response.raise_for_status()
            results = response.json()

            if "results" in results:
                web_results = [result["content"] for result in results["results"][:3]]
                logger.info(f"Retrieved {len(web_results)} web search results.")
                return web_results
            else:
                logger.warning("No web search results found.")
                return []

        except Exception as e:
            logger.error(f"Web search error: {e}", exc_info=True)
            return []
    
    def generate_response(self, query: str, temple_id: str, top_k: int = 3):
        logger.info(f"Generating response for query: '{query}' with temple_id: '{temple_id}' and top_k: {top_k}")

        documents = self.retrieve_relevant_documents(query, temple_id, top_k)
        
        if not documents:
            logger.info(f"No relevant documents found locally for '{query}', using web search.")
            web_results = self.search_web(query)
            logger.info(f"Webs search results: {web_results}")
            if not web_results:
                return {"response": "No relevant information found locally or online."}
            context = "\n".join(web_results)
        else:
            # If documents are found, still use web search for additional context
            combined_results = documents
            context = "\n".join(combined_results)

        logger.info(f"Using {len(context.splitlines())} lines of context for response generation.")

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
