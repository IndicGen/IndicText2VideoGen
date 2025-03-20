import requests
from bs4 import BeautifulSoup
import json
import re

url = "https://kalyangeetha.wordpress.com/2022/02/17/must-visit-ancient-central-karnataka-temples-part-4-hoysala-temples-trail/"
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

temples={}
keywords=["temple","kovil","Mandir","Devasthan","devdhanam","devadhanam"]
for key,value in data.items():
    for word in keywords:
        if word.lower() in key.lower() and "temples" not in key.lower():
            temples[key]=value
            break

for name,details in temples.items():
    print(name)

# # Save to JSON file
# with open('temples_filtered.json', 'w', encoding='utf-8') as f:
#     json.dump(temples, f, ensure_ascii=False, indent=4)

# print("Data extracted and saved to temples_filtered.json")


