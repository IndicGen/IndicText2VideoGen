from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.flow import BlogPostFlow
from src.audio_crew.crew import NarrationCrew
from utils.vectorstore import VectorStoreHandler
from src.rag_crew.rag import GeneralRag, TempleRag
from utils.logger_config import logger
from utils.data_handler import DataPreProcessor

import json
import os
import time
import uuid
from dotenv import load_dotenv
from smallest import Smallest

load_dotenv()
smallest_api_key = os.getenv("SMALLEST_API_KEY")

# Global variables to track API calls
api_call_count = 0
last_api_call_time = 0

def synthesize_tts(api_key, text, temple_name, voice="raman", speed=1.0, sample_rate=24000):
    global api_call_count, last_api_call_time

    if not text.strip():
        raise ValueError("Text cannot be empty for TTS synthesis.")

    try:
        client = Smallest(api_key=api_key)
        output_file = f"E:/Coding/OpenSource/blog_post/git_repo/IndicText2VideoGen/audio_output/{temple_name}.wav"
        
        client.synthesize(
            text,
            save_as=output_file,
            voice_id=voice,
            speed=speed,
            sample_rate=sample_rate
        )

        # Update API call count
        api_call_count += 1

        # Introduce a 1-minute delay after every 4 API calls
        if api_call_count % 4 == 0:
            print("Rate limit: Waiting for 60 seconds before the next batch of API calls...")
            time.sleep(60)

        return output_file

    except Exception as e:
        if "Rate Limited" in str(e):
            raise ValueError("Rate limited by TTS API. Please wait and retry.")
        raise ValueError(f"TTS Synthesis failed: {e}")



app = FastAPI()
blog_flow = BlogPostFlow()
narration = NarrationCrew()


class BlogInput(BaseModel):
    blog_url: str


@app.post("/blogpost_content")
async def blogpost_content(lead: BlogInput):
    """Generate summaries for the temples in the blog post given the blog URL."""
    try:
        logger.info(f"Starting blog post summary generation for URL: {lead.blog_url}")
        blog_flow.state["blog_url"] = lead.blog_url

        result = await blog_flow.kickoff_async()

        if not result:
            logger.warning(
                f"Blog post summary generation failed for URL: {lead.blog_url}"
            )
            raise HTTPException(
                status_code=500, detail="Failed to generate blog post summary."
            )
        logger.info(
            f"Successfully completed for URL: {lead.blog_url}"
        )
        return result
    except HTTPException as http_err:
        raise http_err  # Let FastAPI handle known HTTP errors
    except Exception as e:
        logger.error(
            f"Error generating blog post summary for URL '{lead.blog_url}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Internal server error. Please try again later."
        )


class NarrationInput(BaseModel):
    text: str

@app.post("/narration_test")
async def test_narration(lead: NarrationInput):
    crew_input = {"text": lead.text}
    result = await NarrationCrew().crew().kickoff_async(inputs=crew_input)
    
    list_chunks = json.loads(result.raw)
    
    logger.info(f"Generating the voices for the given temple ")
    for i in range(len(list_chunks)):
        synthesize_tts(smallest_api_key,list_chunks[i],f"jambu_{i}")
    logger.info("Voices are now generated")

    return {
        "message":"Voices are successfully generated."
    }

# class DataInput(BaseModel):
#     blog_url: str
#     log_complete: bool
#     log_processed: bool

# @app.post("/create_chatbot_dataset")
# async def create_data_set(lead: DataInput):
#     """Store the extracted data into the vector database for future usage."""
#     try:
#         logger.info(f"Extracting temple data from blog URL: {lead.blog_url}")

#         # Pre processing the data to extract rag worthy information
#         data_preprocessor = DataPreProcessor(url=lead.blog_url)
#         temples = await data_preprocessor.processed_data(
#             log_complete=lead.log_complete, log_processed=lead.log_processed
#         )

#         if not temples:
#             logger.warning(f"No temple data extracted from URL: {lead.blog_url}")
#             raise HTTPException(
#                 status_code=400,
#                 detail="No temple data could be extracted from the provided blog URL.",
#             )

#         logger.info(f"Successfully extracted {len(temples)} temples from the blog.")

#         return {
#             "message": f"Successfully extracted and cleaned{len(temples)} temples.",
#             "temples": temples,
#         }

#     except HTTPException as http_err:
#         raise http_err  # Let FastAPI handle known HTTP errors
#     except Exception as e:
#         logger.error(
#             f"Error extracting temple data from URL '{lead.blog_url}': {e}",
#             exc_info=True,
#         )
#         raise HTTPException(
#             status_code=500, detail="Internal server error. Please try again later."
#         )


# @app.get("/view_all")
# async def view_all():
#     vector_store = VectorStoreHandler(collection_name="temples")
#     processed_documents = vector_store.collection.get()
#     raw_documents = vector_store.complete_collection.get()
#     return {"processed_documents": processed_documents, "raw_documents": raw_documents}


# @app.get("/view_documents")
# async def get_documents(temple_name: str):
#     """Fetch all chunks stored in ChromaDB for a given temple_name."""
#     vector_store = VectorStoreHandler(collection_name="temples")
#     try:
#         logger.info(f"Fetching documents for temple_name: '{temple_name}'")
#         documents = vector_store.get_documents(case_id=temple_name)
#         if not documents:
#             logger.warning(f"No documents found for temple_name: '{temple_name}'")
#             raise HTTPException(
#                 status_code=404, detail=f"No documents found for '{temple_name}'."
#             )

#         return {
#             "case_id": temple_name,
#             "total_documents": len(documents),
#             "documents": documents,
#         }
#     except HTTPException as http_err:
#         raise http_err  # Let FastAPI handle known HTTP errors
#     except Exception as e:
#         logger.error(
#             f"Error fetching documents for temple_name '{temple_name}': {e}",
#             exc_info=True,
#         )
#         raise HTTPException(
#             status_code=500, detail="Internal server error. Please try again later."
#         )


# # This is a side feature, might consider adding in the future.
# class RagInput(BaseModel):
#     question: str


# @app.post("/Chatbot")
# async def chat_bot(lead: RagInput):
#     """Use this endpoint to chat with the model for general queries about the temples."""
#     rag = GeneralRag()
#     try:
#         logger.info(f"Received query: '{lead.question}'")
#         answer = rag.generate_response(query=lead.question)

#         if not answer or (
#             isinstance(answer, dict)
#             and "response" in answer
#             and answer["response"] == "No relevant documents found."
#         ):
#             logger.warning(f"No relevant documents found for query: '{lead.question}'")
#             raise HTTPException(status_code=404, detail="No relevant documents found.")

#         return {"query": lead.question, "response": answer}
#     except HTTPException as http_err:
#         raise http_err  # Let FastAPI handle known HTTP errors

#     except Exception as e:
#         logger.error(f"Unexpected error in /rag endpoint: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500, detail="Internal server error. Please try again later."
#         )


# class TempleRagInput(BaseModel):
#     temple_id: str
#     question: str


# @app.post("/Temple_Chatbot")
# async def temple_chat_bot(lead: TempleRagInput):
#     """Use this endpoint to chat with the model to know more about a specific temples.
#     Provide the temple name to ask queries."""
#     rag = TempleRag()
#     try:
#         logger.info(f"Received query: '{lead.question}'")
#         answer = rag.generate_response(query=lead.question, temple_id=lead.temple_id)

#         if not answer or (
#             isinstance(answer, dict)
#             and "response" in answer
#             and answer["response"] == "No relevant documents found."
#         ):
#             logger.warning(f"No relevant documents found for query: '{lead.question}'")
#             raise HTTPException(status_code=404, detail="No relevant documents found.")

#         return {"query": lead.question, "response": answer}
#     except HTTPException as http_err:
#         raise http_err  # Let FastAPI handle known HTTP errors

#     except Exception as e:
#         logger.error(f"Unexpected error in /rag endpoint: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500, detail="Internal server error. Please try again later."
#         )
