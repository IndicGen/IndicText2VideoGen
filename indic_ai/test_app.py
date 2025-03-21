from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import re
from utils.embedder import EmbeddingHandler
from utils.vectorstore import VectorStoreHandler

app = FastAPI()

# Initialize handlers globally to reuse across requests
embedding_handler = EmbeddingHandler()
vector_store_handler = VectorStoreHandler(db_path="./chroma_db", collection_name="temples")

KEYWORDS = ["temple", "kovil", "mandir", "devasthan", "devdhanam", "devadhanam"]

class UploadRequest(BaseModel):
    url: str

class RAGQueryRequest(BaseModel):
    temple_name: str

# Utility functions for scraping and filtering temple data
def extract_data_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    data = {}

    for header in soup.find_all(re.compile('^h[1-6]$')):
        heading = header.get_text(strip=True)
        content = ""
        next_node = header.find_next_sibling()

        while next_node and not (next_node.name and re.match(r'h[1-6]', next_node.name)):
            if next_node.name:
                content += next_node.get_text(separator=" ", strip=True) + " "
            next_node = next_node.find_next_sibling()

        data[heading] = content.strip()
    return data

def filter_temple_data(data, keywords):
    temples = {}
    for key, value in data.items():
        if any(word.lower() in key.lower() for word in keywords) and "temples" not in key.lower():
            key= re.sub(r"^\d+(\.\d+)?\.?\s*", "", key)
            temples[key] = value
    
    return temples

@app.post("/temple_names")
async def view_temple_name(request: UploadRequest):
    try:
        raw_data = extract_data_from_url(request.url)
        temple_data = filter_temple_data(raw_data, KEYWORDS)

        if not temple_data:
            raise HTTPException(status_code=404, detail="No temples found on the provided URL.") 
        return {"temple_names": temple_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# # Endpoint to scrape, embed, and upload temple data to ChromaDB
# @app.post("/upload_temple_data")
# async def upload_temple_data(request: UploadRequest):
#     try:
#         raw_data = extract_data_from_url(request.url)
#         temple_data = filter_temple_data(raw_data, KEYWORDS)

#         if not temple_data:
#             raise HTTPException(status_code=404, detail="No temples found on the provided URL.")

#         for temple_name, details in temple_data.items():
#             vector_store_handler.add_text(case_id=temple_name, text=details)

#         return {"status": "success", "message": f"{len(temple_data)} temples uploaded successfully."}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Endpoint to perform RAG queries based on temple name
# @app.post("/rag_query")
# async def rag_query(request: RAGQueryRequest):
#     try:
#         results = vector_store_handler.get_documents(
#             case_id=request.temple_name)

#         if not results:
#             raise HTTPException(status_code=404, detail="No relevant documents found.")

#         return {
#             "temple_name": request.temple_name,
#             "results": results
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
