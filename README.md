# IndicText2VideoGen

This branch of IndicText2VideoGen is an attempt at designing a multi-agent system designed to generate short YouTube videos from blog content. 
It extracts structured information from a given blog URL, generates a script, and processes the content using CrewAI agents.

## Project Structure

```
IndicText2VideoGen/
│── chroma_db/                  # Directory for ChromaDB-related storage
│── indic_ai/                   # Main package for AI processing
│   ├── chroma_db/              # Additional ChromaDB-related modules
│   ├── config/                 # Configuration files
│   │   ├── env.py              # Env.py file which accesses the .env file 
│   ├── src/                    # Source code for various agents
│   │   ├── audio_crew/         # CrewAI agents for generating audio from the content
│   │   ├── content_crew/       # CrewAI agents for content extraction & summarization
│   │   │   ├── config/         # Configuration files for content processing
│   │   │   │   ├── agents.yaml # YAML config for CrewAI agents
│   │   │   │   ├── configs.py  # Python-based configurations
│   │   │   │   ├── tasks.yaml  # YAML config for task definitions
│   │   ├── tools/              # Custom tools for CrewAI
│   │   ├── __init__.py         # Package initialization
│   │   ├── crew.py             # Creating the crew for content creation 
│   │   ├── flow.py             # Main flow execution file for different crews
│   ├── utils/                  # Utility functions
│   │   ├── embedder.py         # Embedding-related utilities
│   │   ├── name_extractor.py   # Extracts temple names and the contents given the blog url
│   │   ├── vectorstore.py      # Vector database Handler
│── app.py                      # Main application entry point
│── .env                        # Environment variables (excluded in .gitignore)
│── .gitignore                  # Git ignore file
│── requirements.txt             # Python dependencies
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repo-url>
cd IndicText2VideoGen
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
- Python 3.7+
- Libraries: Install the required libraries using the following:

  ```bash
  pip install -r requirements.txt
  ```
  #### Required Libraries
  ```bash
  - crewai
  - fastapi
  - litellm
  - bs4
  - requests
  - chromadb
  - langchain
  - langchain-community
  ```
### 4. Configure Environment Variables
Create a `.env` file in the root directory and add necessary API keys and configurations.

### 5. Running the Application
Move to `indic_ai` directory in the terminal and use the following to run the main script:
```bash
uvicorn app:app
```
- Utilize the `blog_test` endpoint to generate the content, where the input is a json with  `blog_url` and `vector_log`.
- `blog_url` is of type string, provide the url of the blog to generate the content
- `vector_log` is of type boolean, provide True to log the contents of the blog into vector store, to chat with the content.


## Features
- **Text-to-Script Conversion:** Extracts structured blog content and generates a video script.
- **Multi-Agent Processing:** Uses CrewAI agents for different tasks (audio, content extraction, summarization).
- **Retrieval-Augmented Generation (RAG):** Chat with the content of the blog to know more about the temple ( under development )

## Contributing
Feel free to submit issues and pull requests!

## License
MIT License

