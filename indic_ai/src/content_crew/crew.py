# src/research_crew/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import os

@CrewBase
class ContentCrew():
    """Content creation crew"""
    
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