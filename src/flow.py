from dotenv import load_dotenv
import litellm
import os
from pydantic import BaseModel
from crewai.flow import Flow, start, listen
from src.content_crew.crew import ContentCrew
from utils.name_extractor import get_data

litellm.api_key = os.getenv("NVIDIA_API_KEY")
litellm.api_base = os.getenv("NVIDIA_LLM_ENDPOINT")


class LeadInput(BaseModel):
    blog_url: str


class blogpostflow(Flow):
    @start()
    def fetch_lead(self):
        blog_url = self.state.get("blog_url", "default_blog_url")
        self.state["blog_url"] = blog_url
        return {"message:": "blog_url is fetched"}

    @listen(fetch_lead)
    def create_data(self):
        names = get_data(url=self.state["blog_url"])
        self.state["names"] = names
        return {"message": "list with temple names is created"}

    @listen(create_data)
    async def run_blog(self):
        content_crew_inputs = {"data": self.state["names"]}
        blog = await ContentCrew().crew().kickoff_async(inputs=content_crew_inputs)
        return blog.raw
