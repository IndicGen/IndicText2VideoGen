from dotenv import load_dotenv

load_dotenv()

import asyncio
import litellm
import os, re, json
from crewai.flow import Flow, start, listen
from asyncio import Semaphore, gather

from src.content_crew.crew import ContentCrew
from src.audio_crew.crew import NarrationCrew
from src.audio_crew.utils.tts_util import TTSProcessor

from utils.data_handler import DataPreProcessor
from utils.logger_config import logger
from utils.vectorstore import VectorStoreHandler, StoreUtils

litellm.api_key = os.getenv("NVIDIA_NIM_API_KEY")
litellm.api_base = os.getenv("NVIDIA_LLM_ENDPOINT")


class ExtractFlow(Flow):
    @start()
    def fetch_lead(self):
        logger.info("Fetching blog URL.")
        blog_url = self.state.get("blog_url", "default_blog_url")
        self.state["blog_url"] = blog_url
        logger.info(f"Blog URL set: {blog_url}")
        return {"message": "blog_url is fetched"}

    @listen(fetch_lead)
    def create_data(self):
        logger.info("Extracting temple data from blog URL.")
        data_preprocessor = DataPreProcessor(url=self.state["blog_url"])
        temples = data_preprocessor.extract_data()
        self.state["temples"] = temples
        logger.info(f"Extracted {len(temples)} temples from the blog.")
        return {"message": "Data extraction from blog is complete"}

    @listen(create_data)
    async def summarize_blog(self):
        logger.info("Processing temple descriptions asynchronously.")
        temple_data = self.state["temples"]

        async def process_temple(temple_name, description):
            logger.info(f"Processing temple: {temple_name}")
            crew_input = {"data": description}
            result = await ContentCrew().crew().kickoff_async(inputs=crew_input)

            tts_processor = TTSProcessor()
            cleaned_summary = tts_processor.clean_text(result.raw)
            logger.info(f"Processing completed for temple: {temple_name}")

            return temple_name, cleaned_summary

        tasks = [process_temple(name, desc) for name, desc in temple_data.items()]
        results = await asyncio.gather(*tasks)

        summaries = {name: summary for name, summary in results}
        logger.info("All temple descriptions processed successfully.")

        self.state["summaries"] = summaries
        return {"message": "Summarizing blog is complete"}

    @listen(summarize_blog)
    async def generate_tts_text(self):
        logger.info("Processing the temples for audio generation asynchronously.")
        temple_summaries = self.state["summaries"]

        async def process_temple(temple_name, description):
            logger.info(f"Processing temple: {temple_name}")
            crew_input = {"text": description}
            result = await NarrationCrew().crew().kickoff_async(inputs=crew_input)
            logger.info(f"Processing completed for temple: {temple_name}")

            entries = re.split(r"\n?\d+\.\s", result.raw.strip())
            final_list = [entry.strip() for entry in entries if entry.strip()]

            return temple_name, final_list

        tasks = [process_temple(name, desc) for name, desc in temple_summaries.items()]
        results = await asyncio.gather(*tasks)

        tts_lists = {name: lists for name, lists in results}
        logger.info("All the temple descriptiosn are ready for TTS")

        self.state["tts_lists"] = tts_lists
        return {
            "message": "Generating texts for TTS is complete",
            "tts_lists": tts_lists,
        }

    @listen(generate_tts_text)
    async def store_in_db(self):
        logger.info("Generating the voice-over for each temple")
        temple_dict = self.state["tts_lists"]
        vector_store = VectorStoreHandler(collection_name="TTS_collection")
        store_utils = StoreUtils(vector_store_handler=vector_store)

        for name, details in temple_dict.items():
            for info in details:
                vector_store.add_text(case_id=name, text=info)
        
        store_utils.add_temple_embeddings()
        return {"message": "Uploaded the details to vectordb"}
