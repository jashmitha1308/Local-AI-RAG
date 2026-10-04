from pathlib import Path
import hashlib

import chromadb
import ollama
import streamlit as st
from pypdf import PdfReader

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
DOCUMENTS_FOLDER = Path("documents")
UPLOADS_FOLDER = DOCUMENTS_FOLDER / "uploads"


def split_into_chunks(text, size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start : start + size])
        if chunk:
            chunks.append(chunk)
        start += size - overlap

    return chunks


@st.cache_data(show_spinner=False)
def get_embedding(text):
    result = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return result["embedding"]


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="data/chroma")
    return client.get_or_create_collection(
        name="local_ai_documents",
        metadata={"hnsw:space": "cosine"},
    )


st.title("Local AI Knowledge Retrieval")

if "messages" not in st.session_state:
    st.session_state.messages = []

UPLOADS_FOLDER.mkdir(parents=True, exist_ok=True)

uploaded_files = st.file_uploader(
    "Upload PDFs or text files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

for uploaded_file in uploaded_files:
    safe_name = Path(uploaded_file.name).name
    saved_path = UPLOADS_FOLDER / safe_name
    saved_path.write_bytes(uploaded_file.getvalue())

texts = []

text_files = {file.name: file for file in DOCUMENTS_FOLDER.glob("*.txt")}
text_files.update({
    file.name: file
    for file in UPLOADS_FOLDER.glob("*.txt")
})

for name, text_file in text_files.items():
    texts.append((
        name,
        text_file.read_text(encoding="utf-8", errors="replace"),
    ))

pdf_files = {file.name: file for file in DOCUMENTS_FOLDER.glob("*.pdf")}
pdf_files.update({
    file.name: file
    for file in UPLOADS_FOLDER.glob("*.pdf")
})

for pdf_path in pdf_files.values():
    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        source = f"{pdf_path.name} (page {page_number})"
        texts.append((source, page_text))

chunks = []

for source, text in texts:
    for chunk_number, chunk_text in enumerate(split_into_chunks(text)):
        chunk_id = hashlib.sha256(
            f"{source}|{chunk_number}|{chunk_text}".encode("utf-8")
        ).hexdigest()

        chunks.append({
            "id": chunk_id,
            "source": source,
            "text": chunk_text,
        })

collection = get_collection()

if chunks:
    sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))

    for source in sources:
        collection.delete(where={"source": source})

    embeddings = [get_embedding(chunk["text"]) for chunk in chunks]

    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[{"source": chunk["source"]} for chunk in chunks],
    )

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask about your documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    if chunks:
        question_embedding = get_embedding(question)
        sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(3, len(chunks)),
            where={"source": {"$in": sources}},
        )

        passages = results["documents"][0]
        result_sources = results["metadatas"][0]

        context = "\n\n".join(
            f"Source: {metadata['source']}\n{passage}"
            for passage, metadata in zip(passages, result_sources)
        )

        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        'Answer only using the provided passages. If the answer '
                        'is missing, say "I couldn’t find that in the documents."\n\n'
                        f"Passages:\n{context}"
                    ),
                },
                *st.session_state.messages,
            ],
        )

        answer = response["message"]["content"]
        source_names = ", ".join(
            dict.fromkeys(metadata["source"] for metadata in result_sources)
        )
        answer += f"\n\nSources: {source_names}"
    else:
        answer = "Upload a text file or PDF first."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()