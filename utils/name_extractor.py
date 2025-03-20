from bs4 import BeautifulSoup
import requests
import json
import re
from typing import List
from utils.vectorstore import VectorStoreHandler
from src.content_crew.config.configs import keywords

vector_store_handler=VectorStoreHandler()

def get_data(url)-> List[str]:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    data = {}
    # Iterate over all headings (h1 to h6)
    for header in soup.find_all(re.compile('^h[1-6]$')):
        heading = header.get_text(strip=True)
        content = ""
        next_node = header.find_next_sibling()

        # Collect content until the next heading
        while next_node and not (next_node.name and re.match(r'h[1-6]', next_node.name)):
            if next_node.name:
                content += next_node.get_text(separator=" ", strip=True) + " "
            next_node = next_node.find_next_sibling()

        data[heading] = content.strip()
    
    # Filtering out the content present only on temples
    temples={}
    for key,value in data.items():
        for word in keywords:
            if word.lower() in key.lower() and "temples" not in key.lower():
                temples[key]=value
                break
    
    # Uploading the details of the temples to chromadb
    for temple_name, details in temples.items():
        print(f"[DEBUG] Content about {temple_name} with length {len(details)} is being uploaded")
        vector_store_handler.add_text(case_id=temple_name, text=details)
    
    temple_names=[]
    for name,details in temples.items():
        temple_names.append(name)
    
    return temple_names
