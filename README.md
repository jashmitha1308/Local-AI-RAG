# Local AI Knowledge Retrieval

Ask questions about your text files and PDFs using local AI.

## Requirements

- Python
- Ollama

## Setup on Windows

Open a terminal in this project folder and run:

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text