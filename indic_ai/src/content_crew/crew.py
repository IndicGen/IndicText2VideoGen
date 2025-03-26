import logging
import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from utils.logger_config import logger

@CrewBase
class ContentCrew:
    """Content creation crew"""
    
    @agent
    def temple_tour_guide_agent(self) -> Agent:
        logger.info("Initializing temple_tour_guide_agent...")
        agent = Agent(
            config=self.agents_config['temple_tour_guide_agent'],
            verbose=True,
            llm=os.getenv("LLM_MODEL")    
        )
        logger.info("temple_tour_guide_agent initialized successfully.")
        return agent
        
    @task
    def temple_tour_guide_task(self) -> Task:
        logger.info("Creating temple_tour_guide_task...")
        task = Task(
            config=self.tasks_config['temple_tour_guide_task'],
        )
        logger.info("temple_tour_guide_task created successfully.")
        return task

    @crew
    def crew(self) -> Crew:
        """Creates the research crew"""
        logger.info("Assembling ContentCrew with agents and tasks...")
        crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
        logger.info("ContentCrew initialized successfully.")
        return crew
