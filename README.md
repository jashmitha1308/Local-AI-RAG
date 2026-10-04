# Local AI Knowledge Retrieval

A local question-answering app for text files and PDFs.

## Features

- Search `.txt` files in `documents`
- Upload multiple PDFs
- Keep uploaded PDFs in `documents` for later runs
- Search relevant passages using Ollama embeddings and Chroma
- Show source filenames and PDF page numbers

## Requirements

- Python
- Ollama

## Setup on Windows

Open a terminal in the project folder and run:

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text