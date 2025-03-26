import requests
import json
import re
from bs4 import BeautifulSoup
from typing import List
from utils.vectorstore import VectorStoreHandler
from src.content_crew.config.configs import keywords
from utils.logger_config import logger

vector_store_handler = VectorStoreHandler()

def get_data(url, log) -> List[str]:
    logger.info(f"Fetching data from URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL: {str(e)}")
        return {}
    
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
    
    temples = {}
    for key, value in data.items():
        for word in keywords:
            if word.lower() in key.lower() and "temples" not in key.lower():
                key = re.sub(r"^\d+(\.\d+)?\.?\s*", "", key)
                temples[key] = value
                break
    
    if log:
        for temple_name, details in temples.items():
            logger.info(f"Uploading content about {temple_name} with length {len(details)}")
            vector_store_handler.add_text(case_id=temple_name, text=details)
    
    logger.info("Data extraction completed successfully.")
    return temples
