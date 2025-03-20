# src/research_crew/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.content_crew.tools.extract_tool import ExtractorTool
import litellm,os

litellm.api_key = os.getenv("NVIDIA_API_KEY")
litellm.api_base = os.getenv("NVIDIA_LLM_ENDPOINT")

@CrewBase
class ContentCrew():
    """Research crew for comprehensive topic analysis and reporting"""
    
    @agent
    def temple_tour_guide_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['temple_tour_guide_agent'],
            verbose=True,
            llm=os.getenv("LLM_MODEL")
            
        )
        
    @task
    def temple_tour_guide_task(self) -> Task:
        return Task(
            config=self.tasks_config['temple_tour_guide_task'],
            tools=[ExtractorTool()]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the research crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )