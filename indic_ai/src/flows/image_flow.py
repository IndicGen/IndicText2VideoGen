from dotenv import load_dotenv

load_dotenv()

import litellm
import os, asyncio
from crewai.flow import Flow, start, listen

from src.image_crew.crew import ImageCrew

from utils.scene_utils import ChunkHandler
from utils.vectorstore import VectorStoreHandler
from utils.logger_config import logger

litellm.api_key = os.getenv("NVIDIA_NIM_API_KEY")
litellm.api_base = os.getenv("NVIDIA_LLM_ENDPOINT")


class ImageFlow(Flow):
    @start()
    def fetch_lead(self):
        logger.info("Fetching temple name.")

        temple_name = self.state.get("temple_name", "default_temple_name")
        self.state["temple_name"] = temple_name

        logger.info(f"Temple_name set: {temple_name}")
        return {"message": "temple_name is fetched"}

    @listen(fetch_lead)
    def fetch_data(self):
        logger.info("Retrieving temple data from Db")

        vector_store = VectorStoreHandler(collection_name="TTS_collection")
        self.state["temple_docs"] = vector_store.get_tts_documents(
            case_id=self.state["temple_name"]
        )

        logger.info(f"Retrieved temple data from Db")

        return {
            "message": "Data retrieval from db is complete",
            "temple_docs": self.state["temple_docs"],
        }

    @listen(fetch_data)
    def chunk_merging(self):
        logger.info("Combining the chunks into one coherent passage")
        
        merger = ChunkHandler()
        self.state["merged_docs"] = merger.merge(self.state["temple_docs"])
        
        return {
            "message": "Chunks from the Db are now converted to single passage",
            "temple_name": self.state["temple_name"],
        }

    @listen(chunk_merging)
    async def generate_prompts(self):
        temple_name = self.state["temple_name"]
        logger.info(f"Processing temple: {temple_name}")
        desc = self.state["merged_docs"]

        crew_input = {"data": desc}

        results = await ImageCrew().crew().kickoff_async(inputs=crew_input)

        return {
            "temple_name": self.state["temple_name"],
            "image_descripts": results.raw,
        }
    
    # @listen(generate_prompts)
    # async def generate_images(self):
    #     pass
