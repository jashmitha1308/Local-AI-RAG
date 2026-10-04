from pathlib import Path
import hashlib
import re

import chromadb
import ollama
import streamlit as st
from pypdf import PdfReader


# ---------- Settings ----------
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

DOCUMENTS_DIR = Path("documents")
UPLOADS_DIR = DOCUMENTS_DIR / "uploads"
CHROMA_DIR = Path("data") / "chroma"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RESULT_COUNT = 5

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Chroma database ----------
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


client = get_chroma_client()
collection = client.get_or_create_collection(name="local_ai_documents")


# ---------- Read and split documents ----------
def split_into_chunks(text):
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def read_document(path):
    """Return document text sections, with PDF page numbers when available."""
    sections = []

    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        sections.append((text, None))

    elif path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                sections.append((text, page_number))

    return sections


def make_id(source, page_number, chunk_number, text):
    unique_text = f"{source}|{page_number}|{chunk_number}|{text}"
    return hashlib.sha256(unique_text.encode("utf-8")).hexdigest()


# ---------- Build the document index ----------
def rebuild_index():
    # Remove old records so deleted or changed files don't leave stale text behind.
    old_records = collection.get(include=[])
    old_ids = old_records.get("ids", [])

    if old_ids:
        collection.delete(ids=old_ids)

    files = [
        path
        for path in DOCUMENTS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}
    ]

    if not files:
        return 0

    records = {}

    for path in files:
        source = path.relative_to(DOCUMENTS_DIR).as_posix()

        try:
            sections = read_document(path)
        except Exception as error:
            st.warning(f"Could not read {source}: {error}")
            continue

        for section_text, page_number in sections:
            for chunk_number, chunk in enumerate(split_into_chunks(section_text)):
                chunk_id = make_id(source, page_number, chunk_number, chunk)

                # A dictionary guarantees every ID is unique before Chroma receives it.
                metadata = {"source": source}
                if page_number is not None:
                    metadata["page"] = page_number

                records[chunk_id] = {
                    "text": chunk,
                    "metadata": metadata,
                }

    if not records:
        return 0

    ids = list(records.keys())
    texts = [records[item_id]["text"] for item_id in ids]
    metadatas = [records[item_id]["metadata"] for item_id in ids]
    embeddings = []

    for text in texts:
        result = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        embeddings.append(result["embedding"])

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(ids)


# ---------- Find relevant passages ----------
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "of", "on", "or",
    "our", "please", "that", "the", "their", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with", "would",
}


def question_terms(question):
    words = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
    return {word for word in words if len(word) > 2 and word not in STOP_WORDS}


def find_passages(question):
    total = collection.count()
    if total == 0:
        return []

    query_embedding = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=question,
    )["embedding"]

    # Semantic search
    semantic = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(15, total),
        include=["documents", "metadatas", "distances"],
    )

    candidates = {}

    for index, text in enumerate(semantic["documents"][0]):
        metadata = semantic["metadatas"][0][index]
        distance = semantic["distances"][0][index]
        key = f"{metadata.get('source')}|{metadata.get('page')}|{text}"

        candidates[key] = {
            "text": text,
            "metadata": metadata,
            "distance": distance,
        }

    # Also check all indexed text for exact keyword matches.
    all_records = collection.get(include=["documents", "metadatas"])
    terms = question_terms(question)

    for text, metadata in zip(
        all_records.get("documents", []),
        all_records.get("metadatas", []),
    ):
        text_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))
        overlap = len(terms & text_words)

        if overlap:
            key = f"{metadata.get('source')}|{metadata.get('page')}|{text}"
            if key not in candidates:
                candidates[key] = {
                    "text": text,
                    "metadata": metadata,
                    "distance": 999,
                }

    # Rank passages with matching words first, then use semantic distance.
    def rank(item):
        text_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", item["text"].lower()))
        overlap = len(terms & text_words)
        return (-overlap, item["distance"])

    return sorted(candidates.values(), key=rank)[:RESULT_COUNT]


def direct_answer_from_passages(question, passages):
    """Use a matching sentence if Ollama incorrectly refuses to answer."""
    terms = question_terms(question)
    best_sentence = ""
    best_score = 0

    for passage in passages:
        sentences = re.split(r"(?<=[.!?])\s+", passage["text"])

        for sentence in sentences:
            sentence_words = set(
                re.findall(r"\b[a-zA-Z0-9]+\b", sentence.lower())
            )
            score = len(terms & sentence_words)

            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()

    if terms and best_score >= max(1, len(terms) // 2):
        return best_sentence

    return None


# ---------- Streamlit page ----------
st.set_page_config(page_title="Local AI Knowledge Retrieval")
st.title("Local AI Knowledge Retrieval")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = UPLOADS_DIR / Path(uploaded_file.name).name
        save_path.write_bytes(uploaded_file.getvalue())

    st.success("Files saved. Click **Rebuild document index** next.")

if st.button("Rebuild document index"):
    try:
        with st.spinner("Reading and indexing your documents..."):
            indexed_count = rebuild_index()

        if indexed_count:
            st.success(f"Done. Indexed {indexed_count} text sections.")
        else:
            st.warning("No readable text found. Add a PDF or TXT file first.")

    except Exception as error:
        st.error("Indexing failed. Check that Ollama is running.")
        st.exception(error)

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            passages = find_passages(question)

            if not passages:
                answer = "I couldn’t find that in the documents."
            else:
                context_parts = []
                sources = []

                for passage in passages:
                    metadata = passage["metadata"]
                    source = metadata.get("source", "Unknown file")
                    page = metadata.get("page")

                    citation = (
                        f"{source} (page {page})"
                        if page
                        else source
                    )

                    context_parts.append(
                        f"Source: {citation}\n{passage['text']}"
                    )
                    sources.append(citation)

                context = "\n\n".join(context_parts)

                response = ollama.chat(
                    model=CHAT_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Answer using only the provided passages. "
                                "If a passage contains the answer, answer it "
                                "clearly and briefly. Never say the answer is "
                                "missing when it appears in a passage. "
                                "If none of the passages contain the answer, "
                                "say: I couldn’t find that in the documents."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\n\n"
                                f"Passages:\n{context}"
                            ),
                        },
                    ],
                    options={"temperature": 0},
                )

                answer = response["message"]["content"].strip()

                refusal_phrases = (
                    "couldn’t find",
                    "couldn't find",
                    "cannot find",
                    "not found in",
                    "not mentioned",
                    "does not contain",
                )

                if any(phrase in answer.lower() for phrase in refusal_phrases):
                    direct_answer = direct_answer_from_passages(
                        question, passages
                    )
                    if direct_answer:
                        answer = direct_answer

                answer += (
                    "\n\nSources: "
                    + ", ".join(dict.fromkeys(sources))
                )

                with st.expander("Retrieved passages"):
                    st.text(context)

            st.markdown(answer)

        except Exception as error:
            answer = (
                "I couldn’t connect to Ollama. Check that Ollama is running "
                "and that the models are installed."
            )
            st.error(answer)
            st.caption(str(error))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )