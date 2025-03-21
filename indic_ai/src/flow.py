from dotenv import load_dotenv
load_dotenv()

import asyncio
import litellm
import os,re
from crewai.flow import Flow, start, listen
from src.content_crew.crew import ContentCrew
from utils.name_extractor import get_data

litellm.api_key = os.getenv("NVIDIA_NIM_API_KEY")
litellm.api_base = os.getenv("NVIDIA_LLM_ENDPOINT")

def clean_text_for_tts(text):
    """Cleans and formats text for smooth TTS processing."""
    text = re.sub(r"##\s*", "", text)  # Remove section headers
    text = re.sub(r"\*\*\s*", "", text)  # Remove double asterisks
    text = re.sub(r"\*\s*", "", text)  # Remove single asterisks
    text = re.sub(r"\n+", ". ", text)  # Convert excessive newlines into a single space
    text = re.sub(r"(\d+)\.\s*", r"Point \1. ", text)  # Convert numbered lists to spoken format
    
    # Replace numerical values with words for better pronunciation
    text = text.replace("1008", "one thousand and eight")
    text = text.replace("3500", "three thousand five hundred")
    text = text.replace("57", "fifty-seven")
    
    # Ensure sentence ends properly
    if not text.endswith("."):
        text += "."

    return text

class blogpostflow(Flow):
    @start()
    def fetch_lead(self):
        blog_url = self.state.get("blog_url", "default_blog_url")
        self.state["blog_url"] = blog_url
        return {"message:": "blog_url is fetched"}

    @listen(fetch_lead)
    def create_data(self):
        temples = get_data(url=self.state["blog_url"],log=self.state["vector_log"])
        self.state["temples"] = temples
        return {"message": "list with temple names is created"}

    @listen(create_data)
    async def run_blog(self):
        temple_data = self.state["temples"]
        
        async def process_temple(temple_name, description):
                crew_input= {"data": description}
                result = await ContentCrew().crew().kickoff_async(inputs=crew_input)
                cleaned_summary = clean_text_for_tts(result.raw) 
                return temple_name, cleaned_summary

        tasks = [process_temple(name, desc) for name, desc in temple_data.items()]
        results = await asyncio.gather(*tasks)

        summaries = {name: summary for name, summary in results}

        return {"summaries": summaries}