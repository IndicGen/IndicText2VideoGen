import requests
import re
import asyncio, httpx
from bs4 import BeautifulSoup
from typing import Dict
from utils.vectorstore import VectorStoreHandler,StoreUtils
from src.content_crew.config.configs import keywords
from utils.logger_config import logger
from config.env import NVIDIA_NIM_API_KEY, NVIDIA_LLM_ENDPOINT, RAG_MODEL


class LlmProcessor:
    """
    This class is used to get the summary regarding the temples for chatbot purposes.
    This utilizes the async functionality given by httpx, this ensures multiple llm calls to happen parallely, reducing the time taken for summary generation. 
    - summarize function is used to generate the summary for each temple based on the prompt given. 
    """ 
    def __init__(self):
        logger.info("Initializing LlmProcessor...")
        self.async_client = httpx.AsyncClient(
            base_url=NVIDIA_LLM_ENDPOINT,
            headers={"Authorization": f"Bearer {NVIDIA_NIM_API_KEY}"},
            timeout=600,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=100),
        )

        logger.info("LlmProcessor initialized successfully.")

    async def summarize(self, temple_name, context):
        prompt = f"""
        Temple Name: \n{temple_name}\n
        Context:\n{context}\n
        User Query: 
        \n Extract the following information from the context provided without utilizing your knowledge. Strictly stick to the context provided.\n
            - **Title**: The name of the temple\n
            - **Location**: The exact place where the temple is located, including the city, state, and country.\n
            - **Main Deity**: The primary deity worshiped at the temple.\n
            - **Historical Significance**: A summary of the history of temple, including any important events, rulers, or legends associated with it.\n
            - **Best Time to Visit**: The most suitable time of the year to visit, considering weather, festivals, or specific events.\n
            - **Customs & Rituals**: Important practices, traditions, and rituals performed at the temple, including any visitor guidelines.\n
        Answer:
        """

        try:
            response = await self.async_client.post(
                url="/chat/completions",
                json={
                    "model": RAG_MODEL.strip(),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "top_p": 0.7,
                    "max_tokens": 10000,
                    "stream": False,
                },
            )
            response.raise_for_status()
            json_response = response.json()

            return (
                json_response["choices"][0]["message"]["content"]
                if json_response.get("choices")
                else "Error in LLM response"
            )
        except Exception as e:
            logger.error(
                f"Error during structure summary generation: {e}", exc_info=True
            )
            return "Error in LLM response"


class DataPreProcessor:
    """
    This class is used to clean up the raw data obtained from blog.
    - extract_data is used to get the raw data from the blog
    - processed_data is used to get the cleaned data for RAG purposes
    """
    def __init__(self, url):
        logger.info("Initializing DataPreProcessor...")
        self.url = url
        self.vector_store_handler = VectorStoreHandler()
        self.store_utils = StoreUtils(self.vector_store_handler)
        self.llm = LlmProcessor()
        logger.info("DataPreProcessor initialized successfully.")

    def extract_data(self) -> Dict[str, str]:
        """This function provides the entire data present about the temples in the blog. This is for summary generation, not for Chatbot"""
        logger.info(f"Fetching data from URL: {self.url}")
        try:
            response = requests.get(self.url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL: {str(e)}")
            return {}

        soup = BeautifulSoup(response.content, "html.parser")

        # Extracting the entire blogpost based on the html structure of the blog
        data = {}
        for header in soup.find_all(re.compile("^h[1-6]$")):
            heading = header.get_text(strip=True)
            content = ""
            next_node = header.find_next_sibling()

            while next_node and not (
                next_node.name and re.match(r"h[1-6]", next_node.name)
            ):
                if next_node.name:
                    content += next_node.get_text(separator=" ", strip=True) + " "
                next_node = next_node.find_next_sibling()

            data[heading] = content.strip()

        # Extracting content that is present only in those sections which have the keywords like temple, devasthana, mandir etc..
        temples = {}
        for key, value in data.items():
            for word in keywords:
                if word.lower() in key.lower() and "temples" not in key.lower():
                    key = re.sub(r"^\d+(\.\d+)?\.?\s*", "", key)
                    temples[key] = value
                    break

        logger.info("Data extraction completed successfully.")
        return temples

    async def processed_data(self, log_complete=False,log_processed=False):
        """Asynchronously process temple data using LLM."""
        raw_temples = self.extract_data()
        
        if log_complete:
            """Logging the complete data about the temples into the vectordb into a different collection"""
            logger.info("Complete temple data is being uploaded")
            for temple_name,details in raw_temples.items():
                self.vector_store_handler.add_complete_text(case_id=temple_name,text=details)
                logger.info(f"Complete content about {temple_name} is uploaded")
            
            logger.info("Adding the tepmle name embeddings")
            self.store_utils.add_temple_embeddings(complete=True)
            logger.info("Successfully added the temple name embeddings into complete collection")

        tasks = []
        for temple_name, details in raw_temples.items():
            logger.info(f"Summarizing temple: {temple_name}")
            tasks.append(self.llm.summarize(temple_name, details))

        summaries = await asyncio.gather(*tasks)

        temples = dict(zip(raw_temples.keys(), summaries))

        logger.info("Completed summarization for all temples.")  # Logging completion

        if log_processed:
            """Logging the summary about the temples into the vectordb into a collection"""
            logger.info("Temples' summary is being uploaded to vetordb")
            for temple_name, details in temples.items():
                self.vector_store_handler.add_text(case_id=temple_name, text=details)
                logger.info(f"Content about {temple_name} is uploaded")
            
            # Adding the temple name embeddings the first time the temples are added
            logger.info("Adding the temple name embeddings")
            self.store_utils.add_temple_embeddings(complete= False)
            logger.info("Successfully added the temple name embeddings")

        return temples
