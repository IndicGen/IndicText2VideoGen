from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import os

@CrewBase
class NarrationCrew():
    """Research crew for comprehensive topic analysis and reporting"""
    
    @agent
    def narration_segment_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['narration_segment_agent'],
            verbose=True,
            llm=os.getenv("LLM_MODEL")
            
        )
        
    @task
    def narration_segment_task(self) -> Task:
        return Task(
            config=self.tasks_config['narration_segment_task'],
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