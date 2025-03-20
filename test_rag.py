import requests
from bs4 import BeautifulSoup
import re
import json
from utils.embedder import EmbeddingHandler
from utils.vectorstore import VectorStoreHandler

# Constants
KEYWORDS = ["temple", "kovil", "mandir", "devasthan", "devdhanam", "devadhanam"]
COLLECTION_NAME = "temples"

# Extract data from URL
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

# Filter temples based on keywords
def filter_temple_data(data, keywords):
    temples = {}
    for key, value in data.items():
        if any(word.lower() in key.lower() for word in keywords) and "temples" not in key.lower():
            temples[key] = value
    return temples

# Main workflow execution
def main():
    url = "https://kalyangeetha.wordpress.com/2022/02/17/must-visit-ancient-central-karnataka-temples-part-4-hoysala-temples-trail/"
    
    # Step 1: Extract data from webpage
    raw_data = extract_data_from_url(url)

    # Step 2: Filter temple-related data
    temple_data = filter_temple_data(raw_data, KEYWORDS)

    # Optional: Save filtered data locally for reference
    with open('filtered_temples.json', 'w', encoding='utf-8') as f:
        json.dump(temple_data, f, ensure_ascii=False, indent=4)

    # Step 3: Initialize VectorStoreHandler (uses EmbeddingHandler internally)
    vector_store = VectorStoreHandler(db_path="./chroma_db", collection_name=COLLECTION_NAME)

    # Step 4: Add temple data to ChromaDB using existing modules
    for temple_name, details in temple_data.items():
        print(f"Adding '{temple_name}' to ChromaDB...")
        vector_store.add_text(case_id=temple_name, text=details)

    print("All temples successfully added to ChromaDB!")

    # Example query (RAG retrieval)
    query = "Yoga Narasimha Temple"
    top_k_results = vector_store.search_documents(case_id="Mudigere Sri Yoga Narasimha Swamy Temple", query=query, top_k=3)

    print("\nTop results for query:")
    for idx, doc in enumerate(top_k_results, 1):
        print(f"{idx}. {doc}\n")

if __name__ == "__main__":
    main()
