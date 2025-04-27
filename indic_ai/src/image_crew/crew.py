import logging
import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from utils.logger_config import logger

@CrewBase
class ImageCrew:
    """Content creation crew"""
    
    @agent
    def narrative_analyzer_agent(self) -> Agent:
        logger.info("Initializing narrative_analyzer_agent...")
        agent = Agent(
            config=self.agents_config['narrative_analyzer_agent'],
            verbose=True,
            llm=os.getenv("LLM_MODEL")    
        )
        logger.info("narrative_analyzer_agent initialized successfully.")
        return agent

    @agent
    def image_count_estimator_agent(self) -> Agent:
        logger.info("Initializing image_count_estimator_agent...")
        agent = Agent(
            config=self.agents_config['image_count_estimator_agent'],
            verbose=True,
            llm=os.getenv("LLM_MODEL")    
        )
        logger.info("image_count_estimator_agent initialized successfully.")
        return agent

    @agent
    def prompt_writer_agent(self) -> Agent:
        logger.info("Initializing prompt_writer_agent...")
        agent = Agent(
            config=self.agents_config['prompt_writer_agent'],
            verbose=True,
            llm=os.getenv("LLM_MODEL")    
        )
        logger.info("prompt_writer_agent initialized successfully.")
        return agent

    #=============================================================================

    @task
    def narrative_analyzer_task(self) -> Task:
        logger.info("Creating narrative_analyzer_task...")
        task = Task(
            config=self.tasks_config['narrative_analyzer_task'],
        )
        logger.info("narrative_analyzer_task created successfully.")
        return task
    
    @task
    def image_count_estimator_task(self) -> Task:
        logger.info("Creating image_count_estimator_task...")
        task = Task(
            config=self.tasks_config['image_count_estimator_task'],
        )
        logger.info("image_count_estimator_task created successfully.")
        return task
    
    @task
    def prompt_writer_task(self) -> Task:
        logger.info("Creating prompt_writer_task...")
        task = Task(
            config=self.tasks_config['prompt_writer_task'],
        )
        logger.info("prompt_writer_task created successfully.")
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
