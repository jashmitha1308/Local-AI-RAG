from pathlib import Path
from io import BytesIO

import streamlit as st
import ollama
from pypdf import PdfReader

st.title("Local AI Knowledge Retrieval")

if "messages" not in st.session_state:
    st.session_state.messages = []

texts = [
    file.read_text(encoding="utf-8")
    for file in Path("documents").glob("*.txt")
]

pdf = st.file_uploader("Upload a PDF", type="pdf")
if pdf:
    reader = PdfReader(BytesIO(pdf.getvalue()))
    texts.append("\n".join(page.extract_text() or "" for page in reader.pages))

notes = "\n\n".join(texts)

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask about your documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    if notes:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": (
                        'Answer only from these documents. If the answer is not '
                        'there, say "I couldn’t find that in the documents."\n\n'
                        f"Documents:\n{notes}"
                    ),
                },
                *st.session_state.messages,
            ],
        )
        answer = response["message"]["content"]
    else:
        answer = "Add a text file or upload a PDF first."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()