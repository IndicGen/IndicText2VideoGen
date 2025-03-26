from dotenv import load_dotenv
load_dotenv()

import asyncio
import litellm
import os
import re
from crewai.flow import Flow, start, listen
from src.content_crew.crew import ContentCrew
from utils.name_extractor import get_data
from utils.logger_config import logger

litellm.api_key = os.getenv("NVIDIA_NIM_API_KEY")
litellm.api_base = os.getenv("NVIDIA_LLM_ENDPOINT")

def clean_text_for_tts(text):
    """Cleans and formats text for smooth TTS processing."""
    logger.info("Cleaning text for TTS processing.")
    text = re.sub(r"##\s*", "", text)
    text = re.sub(r"\*\*\s*", "", text)
    text = re.sub(r"\*\s*", "", text)
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"(\d+)\.\s*", r"Point \1. ", text)
    
    text = text.replace("1008", "one thousand and eight")
    text = text.replace("3500", "three thousand five hundred")
    text = text.replace("57", "fifty-seven")
    
    if not text.endswith("."):
        text += "."
    
    logger.info("Text cleaned successfully.")
    return text

class BlogPostFlow(Flow):
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
        temples = get_data(url=self.state["blog_url"], log=self.state["vector_log"])
        self.state["temples"] = temples
        logger.info(f"Extracted {len(temples)} temples from the blog.")
        return {"message": "list with temple names is created"}

    @listen(create_data)
    async def run_blog(self):
        logger.info("Processing temple descriptions asynchronously.")
        temple_data = self.state["temples"]
        
        async def process_temple(temple_name, description):
            logger.info(f"Processing temple: {temple_name}")
            crew_input = {"data": description}
            result = await ContentCrew().crew().kickoff_async(inputs=crew_input)
            cleaned_summary = clean_text_for_tts(result.raw)
            logger.info(f"Processing completed for temple: {temple_name}")
            return temple_name, cleaned_summary

        tasks = [process_temple(name, desc) for name, desc in temple_data.items()]
        results = await asyncio.gather(*tasks)
        
        summaries = {name: summary for name, summary in results}
        logger.info("All temple descriptions processed successfully.")
        return {"summaries": summaries}