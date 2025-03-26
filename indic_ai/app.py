from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Form
from pydantic import BaseModel
from src.flow import BlogPostFlow
from utils.vectorstore import VectorStoreHandler

blog_flow=BlogPostFlow()
app= FastAPI()
        
class LeadInput(BaseModel):
    blog_url : str
    vector_log: bool

@app.post("/test_blogpost")
async def blog_test(lead: LeadInput):
    
    blog_flow.state["blog_url"]=lead.blog_url
    blog_flow.state["vector_log"]=lead.vector_log

    result= await blog_flow.kickoff_async()
    
    return result

# Use this endpoint to see the documents in the given case id
@app.get("/view_chunks/{case_id}")
async def get_documents(case_id: str):
    """Fetch all chunks stored in ChromaDB for a given case_id."""
    vector_store = VectorStoreHandler(collection_name="temples")
    try:
        # Fetch documents based on case_id filter
        case_documents = vector_store.collection.get(where={"case_id": case_id})

        # If no documents found
        if not case_documents or "documents" not in case_documents:
            return {"message": f"No documents found for case_id: {case_id}"}

        # Extracting document texts and metadata
        documents = case_documents.get("documents", [])
        metadata = case_documents.get("metadatas", [])

        return {
            "case_id": case_id,
            "total_documents": len(documents),
            "documents": documents,
            "metadata": metadata,
        }

    except Exception as e:
        return {"error": str(e)}