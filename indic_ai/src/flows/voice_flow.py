from dotenv import load_dotenv

load_dotenv()

import litellm
import os
from crewai.flow import Flow, start, listen

from src.audio_crew.utils.tts_util import TTSProcessor

from utils.vectorstore import VectorStoreHandler
from utils.logger_config import logger

litellm.api_key = os.getenv("NVIDIA_NIM_API_KEY")
litellm.api_base = os.getenv("NVIDIA_LLM_ENDPOINT")


class VoiceFlow(Flow):
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
        vector_store=VectorStoreHandler(collection_name="TTS_collection")
        self.state["temple_docs"]=vector_store.get_tts_documents(case_id=self.state["temple_name"])
        logger.info(f"Retrieved temple data from Db")
        return {"message": "Data retrieval from db is complete", "temple_docs":self.state["temple_docs"]}


    @listen(fetch_data)
    async def generate_voices(self):

        tts_processor = TTSProcessor()
        temple_docs = self.state["temple_docs"]
        temple_name = self.state["temple_name"]
        
        logger.info(f"Generating the voice-over for {temple_name}")
        for i,text in enumerate(temple_docs):
            tts_processor.synthesize_audio_nvidia(text,f"{temple_name}_{i}")
        
        logger.info(f"Stitching the audio files together for: {temple_name}")
        tts_processor.stitch_audio_files(temple_name=temple_name)
        
        return {"message": f"Voice-overs are generated for {temple_name}"}

