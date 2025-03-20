from dotenv import load_dotenv
import litellm
import os
from pydantic import BaseModel
from crewai.flow import Flow, start, listen
from src.crew import BLogPostCrew
from utils.name_extractor import get_data

litellm.api_key = os.getenv("NVIDIA_API_KEY")
litellm.api_base = os.getenv("NVIDIA_LLM_ENDPOINT")


class LeadInput(BaseModel):
    blog_url : str


class blogpostflow(Flow):
    @start()
    def fetch_lead(self):
        blog_url=self.state.get("blog_url","default_blog_url")
        return blog_url

    @listen(fetch_lead)
    def create_data(self,blog_url):
        names=get_data(url=blog_url)
        return {"data": names }

    @listen(create_data)
    async def run_blog(self,lead):
        blog = await BLogPostCrew().crew().kickoff_async(inputs=lead)
        return blog.raw