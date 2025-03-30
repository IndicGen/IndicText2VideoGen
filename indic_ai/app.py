from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.flow import BlogPostFlow
from utils.vectorstore import VectorStoreHandler
from src.rag_crew.rag import GeneralRag, TempleRag
from utils.logger_config import logger
from utils.data_handler import DataPreProcessor

app = FastAPI()
blog_flow = BlogPostFlow()


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
            f"Successfully generated blog post summary for URL: {lead.blog_url}"
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


class DataInput(BaseModel):
    blog_url: str
    log_complete: bool
    log_processed: bool


@app.post("/create_chatbot_dataset")
async def create_data_set(lead: DataInput):
    """Store the extracted data into the vector database for future usage."""
    try:
        logger.info(f"Extracting temple data from blog URL: {lead.blog_url}")

        # Pre processing the data to extract rag worthy information
        data_preprocessor = DataPreProcessor(url=lead.blog_url)
        temples = await data_preprocessor.processed_data(
            log_complete=lead.log_complete, log_processed=lead.log_processed
        )

        if not temples:
            logger.warning(f"No temple data extracted from URL: {lead.blog_url}")
            raise HTTPException(
                status_code=400,
                detail="No temple data could be extracted from the provided blog URL.",
            )

        logger.info(f"Successfully extracted {len(temples)} temples from the blog.")

        return {
            "message": f"Successfully extracted and cleaned{len(temples)} temples.",
            "temples": temples,
        }

    except HTTPException as http_err:
        raise http_err  # Let FastAPI handle known HTTP errors
    except Exception as e:
        logger.error(
            f"Error extracting temple data from URL '{lead.blog_url}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Internal server error. Please try again later."
        )


@app.get("/view_all")
async def view_all():
    vector_store = VectorStoreHandler(collection_name="temples")
    processed_documents = vector_store.collection.get()
    raw_documents = vector_store.complete_collection.get()
    return {"processed_documents": processed_documents, "raw_documents": raw_documents}


@app.get("/view_documents")
async def get_documents(temple_name: str):
    """Fetch all chunks stored in ChromaDB for a given temple_name."""
    vector_store = VectorStoreHandler(collection_name="temples")
    try:
        logger.info(f"Fetching documents for temple_name: '{temple_name}'")
        documents = vector_store.get_documents(case_id=temple_name)
        if not documents:
            logger.warning(f"No documents found for temple_name: '{temple_name}'")
            raise HTTPException(
                status_code=404, detail=f"No documents found for '{temple_name}'."
            )

        return {
            "case_id": temple_name,
            "total_documents": len(documents),
            "documents": documents,
        }
    except HTTPException as http_err:
        raise http_err  # Let FastAPI handle known HTTP errors
    except Exception as e:
        logger.error(
            f"Error fetching documents for temple_name '{temple_name}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Internal server error. Please try again later."
        )


# This is a side feature, might consider adding in the future.
class RagInput(BaseModel):
    question: str


@app.post("/Chatbot")
async def chat_bot(lead: RagInput):
    """Use this endpoint to chat with the model for general queries about the temples."""
    rag = GeneralRag()
    try:
        logger.info(f"Received query: '{lead.question}'")
        answer = rag.generate_response(query=lead.question)

        if not answer or (
            isinstance(answer, dict)
            and "response" in answer
            and answer["response"] == "No relevant documents found."
        ):
            logger.warning(f"No relevant documents found for query: '{lead.question}'")
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        return {"query": lead.question, "response": answer}
    except HTTPException as http_err:
        raise http_err  # Let FastAPI handle known HTTP errors

    except Exception as e:
        logger.error(f"Unexpected error in /rag endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error. Please try again later."
        )


class TempleRagInput(BaseModel):
    temple_id: str
    question: str


@app.post("/Temple_Chatbot")
async def temple_chat_bot(lead: TempleRagInput):
    """Use this endpoint to chat with the model to know more about a specific temples.
    Provide the temple name to ask queries."""
    rag = TempleRag()
    try:
        logger.info(f"Received query: '{lead.question}'")
        answer = rag.generate_response(query=lead.question, temple_id=lead.temple_id)

        if not answer or (
            isinstance(answer, dict)
            and "response" in answer
            and answer["response"] == "No relevant documents found."
        ):
            logger.warning(f"No relevant documents found for query: '{lead.question}'")
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        return {"query": lead.question, "response": answer}
    except HTTPException as http_err:
        raise http_err  # Let FastAPI handle known HTTP errors

    except Exception as e:
        logger.error(f"Unexpected error in /rag endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error. Please try again later."
        )
