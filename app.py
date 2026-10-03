from pathlib import Path
from io import BytesIO
import math

import streamlit as st
import ollama
from pypdf import PdfReader

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"


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


def similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    return dot / (length_a * length_b)


st.title("Local AI Knowledge Retrieval")

if "messages" not in st.session_state:
    st.session_state.messages = []

texts = []
for file in Path("documents").glob("*.txt"):
    texts.append((file.name, file.read_text(encoding="utf-8")))

pdf = st.file_uploader("Upload a PDF", type="pdf")
if pdf:
    reader = PdfReader(BytesIO(pdf.getvalue()))
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    texts.append((pdf.name, pdf_text))

chunks = []
for filename, text in texts:
    for chunk in split_into_chunks(text):
        chunks.append({"source": filename, "text": chunk})

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

        for chunk in chunks:
            chunk["embedding"] = get_embedding(chunk["text"])
            chunk["score"] = similarity(question_embedding, chunk["embedding"])

        best_chunks = sorted(
            chunks,
            key=lambda item: item["score"],
            reverse=True,
        )[:3]

        context = "\n\n".join(
            f"Source: {item['source']}\n{item['text']}"
            for item in best_chunks
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
            dict.fromkeys(item["source"] for item in best_chunks)
        )
        answer += f"\n\nSources: {source_names}"
    else:
        answer = "Add a text file or upload a PDF first."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()