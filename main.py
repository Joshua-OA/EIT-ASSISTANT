import streamlit as st
import chromadb
import os
import uuid
import requests
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Import PyPDF2 for reading PDF files
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Import BeautifulSoup for scraping article content from URLs
try:
    from bs4 import BeautifulSoup
    WEB_SUPPORT = True
except ImportError:
    WEB_SUPPORT = False

st.set_page_config(
    page_title="Joshua's EIT Assistant",
    page_icon="🤖",
    layout="wide",
)

# EIT's personality and behavior instructions passed to OpenAI on every request
EIT_SYSTEM_PROMPT = """You are EIT (Emerging Intelligent Technologist), an AI assistant with a warm,
confident and witty personality — just like your creator.

Your character:
- You call yourself EIT and take pride in it
- You explain things clearly with a bit of swagger
- You are passionate about technology, startups, and African innovation
- When you summarize, you get straight to the point but still add your flavor
- You greet the person before diving into your answer

Always start your response with a short greeting or expression before diving in.
"""

# Initialize ChromaDB in-memory vector store with default embeddings
chroma_client = chromadb.Client()
default_ef = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="eit_knowledge_base",
    embedding_function=default_ef
)

# Pre-load any .txt articles from the data folder into the knowledge base
articles_folder = "./data/new_articles"
if os.path.exists(articles_folder):
    doc_idx = 1
    for filename in os.listdir(articles_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(articles_folder, filename)
            with open(filepath, "r") as f:
                content = f.read().strip()
            if content:
                collection.upsert(ids=f"txt_doc_{doc_idx}", documents=content)
                doc_idx += 1


def extract_text_from_pdf(uploaded_file):
    """Extract all text from an uploaded PDF file, page by page."""
    if not PDF_SUPPORT:
        return None, "PyPDF2 is not installed. Run: pip install PyPDF2"
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        if not full_text.strip():
            return None, "Could not extract readable text from this PDF. Try another file."
        return full_text.strip(), None
    except Exception as e:
        return None, f"Error reading PDF: {str(e)}"


def extract_text_from_url(url):
    """Fetch a webpage and extract the main article text, stripping navigation and ads."""
    if not WEB_SUPPORT:
        return None, "BeautifulSoup is not installed. Run: pip install beautifulsoup4 requests"
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        # Try to isolate the main article body
        article_body = (
            soup.find("article") or
            soup.find("main") or
            soup.find(class_=lambda c: c and any(
                word in str(c).lower() for word in ["article", "content", "post", "story"]
            ))
        )
        target = article_body if article_body else soup.find("body")
        if not target:
            return None, "Could not extract content from this link. Try another URL."

        text = target.get_text(separator="\n", strip=True)
        # Drop short lines that are likely navigation or metadata
        lines = [line for line in text.split("\n") if len(line.strip()) > 40]
        clean_text = "\n".join(lines)

        if not clean_text:
            return None, "The article content was too thin to extract."
        return clean_text[:15000], None  # cap at 15k chars to stay within token limits

    except requests.exceptions.ConnectionError:
        return None, "Could not connect to that URL. Check the link and your internet connection."
    except requests.exceptions.Timeout:
        return None, "The website took too long to respond. Try again."
    except Exception as e:
        return None, f"Something went wrong: {str(e)}"


def add_to_knowledge_base(text, source_label):
    """
    Split text into overlapping chunks and store each one in ChromaDB.
    Chunking ensures large documents fit within embedding limits.
    Returns the number of chunks stored.
    """
    chunk_size = 1000
    overlap = 200
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    for chunk in chunks:
        doc_id = f"{source_label}_{uuid.uuid4().hex[:8]}"
        collection.upsert(ids=doc_id, documents=chunk)
    return len(chunks)


def eit_summarize(text, source_type, model="gpt-4o-mini"):
    """Ask EIT to summarize a piece of text with its personality."""
    user_message = (
        f"Please summarize the following {source_type} content. "
        f"Give me the key points, main arguments, and your take on it:\n\n{text[:8000]}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


def eit_rag_answer(query, chat_history, model="gpt-4o-mini"):
    """
    RAG pipeline:
    1. Retrieve the top 3 relevant chunks from ChromaDB.
    2. Build a prompt that includes those chunks as context.
    3. Replay the full conversation history so EIT remembers prior turns.
    4. Generate and return EIT's answer.
    """
    results = collection.query(query_texts=[query], n_results=3)
    retrieved_docs = results["documents"][0]

    if any(retrieved_docs):
        context = "\n\n---\n\n".join(retrieved_docs)
        user_message = (
            f"Based on the following documents from my knowledge base:\n\n"
            f"{context}\n\n"
            f"Please answer this question: {query}"
        )
    else:
        # No relevant documents found — fall back to general knowledge
        user_message = query

    messages = [{"role": "system", "content": EIT_SYSTEM_PROMPT}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content, results


# ── Session state ──────────────────────────────────────────────────────────────
# Streamlit re-runs the entire script on every interaction.
# session_state persists values across those re-runs.

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_sources" not in st.session_state:
    st.session_state.show_sources = False

if "kb_count" not in st.session_state:
    st.session_state.kb_count = collection.count()

if "link_mode" not in st.session_state:
    st.session_state.link_mode = False

# Track which PDF has already been processed so we don't reprocess on every rerun
if "last_processed_pdf" not in st.session_state:
    st.session_state.last_processed_pdf = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 EIT")
    st.caption("Emerging Intelligent Technologist")
    st.divider()

    selected_model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="gpt-4o-mini is fast and cost-effective. gpt-4o is more powerful."
    )

    st.session_state.show_sources = st.toggle(
        "Show RAG sources",
        value=st.session_state.show_sources,
        help="Show the document chunks EIT retrieved to form each answer."
    )

    st.divider()
    st.metric("Docs in knowledge base", st.session_state.kb_count)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='margin-bottom:0'>🤖 Joshua's EIT — Emerging Intelligent Technologist</h2>",
    unsafe_allow_html=True
)
st.caption(
    "Chat with EIT about anything. "
    "Upload a PDF or click **🔗 Link** to switch the input into URL mode."
)
st.divider()


# ── Chat history display ───────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

        if (
            msg["role"] == "assistant"
            and st.session_state.show_sources
            and msg.get("sources")
            and msg["sources"].get("documents")
        ):
            with st.expander("📚 Sources used"):
                for idx, doc in enumerate(msg["sources"]["documents"][0]):
                    distance = msg["sources"]["distances"][0][idx]
                    doc_id = msg["sources"]["ids"][0][idx]
                    st.caption(f"`{doc_id}` — similarity: {1 - distance:.2f}")
                    st.text(doc[:400] + ("..." if len(doc) > 400 else ""))
                    if idx < len(msg["sources"]["documents"][0]) - 1:
                        st.divider()

# Welcome message shown only on first load
if not st.session_state.chat_history:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            "Hey! I'm **EIT** — Emerging Intelligent Technologist. "
            "Your AI companion for tech, startups, and African innovation. 🚀\n\n"
            "Type a question below, upload a PDF, or paste an article link — I'm here!"
        )


# ── Bottom input area ──────────────────────────────────────────────────────────
# Layout:
#   Row 1:  [ text input .................. ] [ Send ]   ← st.chat_input
#   Row 2:  [ 📄 Upload PDF ]  [ 🔗 Link ]               ← action buttons
#
# st.chat_input() always pins itself to the very bottom of the page.
# We render the action buttons in a fixed-position row just above it using CSS.

st.markdown(
    """
    <style>
    /* Extra bottom padding so messages are never hidden behind the input area */
    .main .block-container {
        padding-bottom: 130px !important;
    }

    /* Pin the column row that contains the file uploader just above the chat bar */
    div[data-testid="stHorizontalBlock"]:has(section[data-testid="stFileUploader"]) {
        position: fixed;
        bottom: 64px;
        left: 50%;
        transform: translateX(-50%);
        width: min(760px, 90vw);
        z-index: 1000;
        background: transparent;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 0;
    }

    /* Strip the file uploader down to just its browse button — no dropzone, no label */
    section[data-testid="stFileUploader"] > div {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    section[data-testid="stFileUploader"] > div > div:first-child {
        display: none !important;
    }
    section[data-testid="stFileUploader"] label {
        display: none !important;
    }
    section[data-testid="stFileUploader"] button {
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Chat input — placeholder changes to signal when link mode is active
if st.session_state.link_mode:
    chat_value = st.chat_input("Paste article URL and press Enter...")
else:
    chat_value = st.chat_input("Ask EIT anything...")

# Action button row — rendered directly after chat_input so CSS can pin it above
upload_col, link_col, _ = st.columns([1.6, 1.4, 7])

with upload_col:
    uploaded_pdf = st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader"
    )

with link_col:
    link_active = st.session_state.link_mode
    if st.button(
        "🔗 Link ✓" if link_active else "🔗 Link",
        type="primary" if link_active else "secondary",
        help="Toggle URL mode — paste an article link directly into the chat input"
    ):
        st.session_state.link_mode = not st.session_state.link_mode
        st.rerun()


# ── Process PDF when a new file is selected ────────────────────────────────────
if uploaded_pdf is not None and uploaded_pdf.name != st.session_state.last_processed_pdf:
    with st.spinner("Reading your PDF..."):
        pdf_text, error = extract_text_from_pdf(uploaded_pdf)

    if error:
        st.error(f"❌ {error}")
    else:
        chunk_count = add_to_knowledge_base(
            pdf_text,
            source_label=f"pdf_{uploaded_pdf.name.replace(' ', '_')}"
        )
        st.session_state.kb_count = collection.count()
        st.session_state.last_processed_pdf = uploaded_pdf.name

        with st.spinner("Summarizing..."):
            summary = eit_summarize(pdf_text, "PDF document", model=selected_model)

        st.session_state.chat_history.append({
            "role": "user",
            "content": f"📄 *Uploaded PDF: **{uploaded_pdf.name}***"
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"**PDF loaded** — {chunk_count} chunks added to my knowledge base.\n\n{summary}",
            "sources": None
        })
        st.rerun()


# ── Process chat input ─────────────────────────────────────────────────────────
if chat_value:
    if st.session_state.link_mode:
        # URL mode: fetch and summarize the article
        with st.spinner("Fetching article..."):
            article_text, error = extract_text_from_url(chat_value)

        if error:
            st.session_state.chat_history.append({
                "role": "user",
                "content": f"🔗 *Link: {chat_value}*"
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"❌ {error}",
                "sources": None
            })
        else:
            domain = chat_value.split("/")[2] if "/" in chat_value else "web"
            chunk_count = add_to_knowledge_base(
                article_text,
                source_label=f"url_{domain}"
            )
            st.session_state.kb_count = collection.count()

            with st.spinner("Summarizing..."):
                summary = eit_summarize(article_text, "web article", model=selected_model)

            st.session_state.chat_history.append({
                "role": "user",
                "content": f"🔗 *Article: {chat_value}*"
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**Article loaded** — {chunk_count} chunks added to my knowledge base.\n\n{summary}",
                "sources": None
            })

        # Return to normal chat mode after the link is processed
        st.session_state.link_mode = False
        st.rerun()

    else:
        # Normal chat mode: send to EIT via RAG
        with st.chat_message("user", avatar="🧑"):
            st.markdown(chat_value)

        st.session_state.chat_history.append({
            "role": "user",
            "content": chat_value
        })

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                answer, sources = eit_rag_answer(
                    chat_value,
                    st.session_state.chat_history[:-1],
                    model=selected_model
                )

            st.markdown(answer)

            if (
                st.session_state.show_sources
                and sources
                and sources.get("documents")
            ):
                with st.expander("📚 Sources used"):
                    for idx, doc in enumerate(sources["documents"][0]):
                        distance = sources["distances"][0][idx]
                        doc_id = sources["ids"][0][idx]
                        st.caption(f"`{doc_id}` — similarity: {1 - distance:.2f}")
                        st.text(doc[:400] + ("..." if len(doc) > 400 else ""))
                        if idx < len(sources["documents"][0]) - 1:
                            st.divider()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
