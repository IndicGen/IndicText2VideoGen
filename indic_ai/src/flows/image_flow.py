from dotenv import load_dotenv

load_dotenv()

import litellm
import random
import re
import os, asyncio
from crewai.flow import Flow, start, listen

from src.image_crew.crew import ImageCrew

from utils.scene_utils import ChunkHandler
from utils.vectorstore import VectorStoreHandler
from utils.logger_config import logger
from litellm import RateLimitError

litellm.api_key = os.getenv("NVIDIA_NIM_API_KEY")
litellm.api_base = os.getenv("NVIDIA_LLM_ENDPOINT")


class ImageFlow(Flow):
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

        vector_store = VectorStoreHandler(collection_name="TTS_collection")
        self.state["temple_docs"] = vector_store.get_tts_documents(
            case_id=self.state["temple_name"]
        )

        logger.info(f"Retrieved temple data from Db")

        return {
            "message": "Data retrieval from db is complete",
            "temple_docs": self.state["temple_docs"],
        }

    @listen(fetch_data)
    async def generate_prompts(self):
        temple_name = self.state["temple_name"]
        logger.info(f"Processing temple: {temple_name}")

        docs = self.state["temple_docs"]

        max_retries = 5
        semaphore = asyncio.Semaphore(1)  # limit concurrency

        results = {}

        async def process_chunk(idx, desc):
            async with semaphore:
                retries = 0
                while retries < max_retries:
                    try:
                        logger.info(f"Processing chunk {idx + 1} for temple: {temple_name}")
                        crew_input = {"data": desc}
                        image_crew = ImageCrew().crew()

                        # Slow down before each request a bit
                        await asyncio.sleep(random.uniform(1.5, 3))

                        result = await image_crew.kickoff_async(inputs=crew_input)
                        results[idx] = result.raw
                        return  # success

                    except RateLimitError as e:
                        retries += 1
                        wait_time = random.uniform(5, 10) * retries
                        logger.warning(f"Rate limit hit at chunk {idx + 1}. Retry {retries}/{max_retries} after {wait_time:.1f} seconds")
                        await asyncio.sleep(wait_time)

                    except Exception as e:
                        logger.error(f"Error processing chunk {idx + 1}: {e}")
                        results[idx] = {"error": str(e)}
                        return

                # After max retries, give up
                results[idx] = {"error": f"Max retries exceeded for chunk {idx + 1}"}


        # Fire all tasks
        tasks = [asyncio.create_task(process_chunk(idx, desc)) for idx, desc in enumerate(docs)]

        await asyncio.gather(*tasks)
        
        def process(text):
            items = re.split(r'\d+\.\s*', text)
            # Remove the first empty string if it appears
            items = [item for item in items if item]
            return items
        
        prompts=[]
        for i,prompt in results.items():
            prompts.append(process(prompt))
        prompt_list=[item for sublist in prompts for item in sublist]
        
        # Storing the prompts in vector store 
        prompt_store=VectorStoreHandler(collection_name="prompts")
        prompt_meta=f"{temple_name}_prompt"
        
        for prompt in prompt_list:
            prompt_store.add_text(case_id=prompt_meta,text=prompt)

        self.state["prompts"] = prompt_list

        return {
            "temple_name": temple_name,
            "image_descripts_per_chunk": prompt_list,
        }