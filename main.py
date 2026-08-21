"""
Qwen3 RAG Chat -- chat with your PDFs using Nebius AI + LlamaIndex.

Upload a PDF in the sidebar and the document is chunked, embedded with
BAAI/bge-en-icl and indexed in memory by LlamaIndex. Questions are answered by
Qwen3-235B-A22B (or DeepSeek-V3) grounded strictly in the retrieved chunks.
"""

import base64
import gc
import hashlib
import os
import re
import tempfile

import streamlit as st
from dotenv import load_dotenv

from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.nebius import NebiusEmbedding
from llama_index.llms.nebius import NebiusLLM

load_dotenv()

APP_TITLE = "Qwen3 RAG Chat"
EMBED_MODEL = "BAAI/bge-en-icl"
MODELS = {
    "Qwen3-235B-A22B": "Qwen/Qwen3-235B-A22B",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
}

QA_PROMPT = PromptTemplate(
    "You are a precise document analyst. Context from the user's document is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using only the context above and no prior knowledge, answer the query. "
    "If the context does not contain the answer, say so plainly instead of guessing. "
    "Be specific, and quote the document's wording where it helps.\n"
    "Query: {query_str}\n"
    "Answer: "
)

st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_secret(name: str) -> str:
    """st.secrets access that tolerates the file being absent (local runs)."""
    try:
        return str(st.secrets.get(name, "") or "").strip()
    except Exception:
        return ""


def resolve_api_key() -> str:
    """Key typed in the sidebar wins, then st.secrets, then the environment."""
    if st.session_state.get("api_key"):
        return st.session_state["api_key"].strip()
    return read_secret("NEBIUS_API_KEY") or (os.getenv("NEBIUS_API_KEY") or "").strip()


def split_reasoning(text: str) -> tuple[str, str]:
    """Separate Qwen3's <think> block from the user-facing answer."""
    thoughts = "\n\n".join(re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)).strip()
    answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # An unterminated block means the stream is still inside the thought.
    if "<think>" in answer:
        head, _, tail = answer.partition("<think>")
        thoughts = (thoughts + "\n\n" + tail).strip()
        answer = head
    return thoughts, answer.strip()


def render_pdf(file_bytes: bytes) -> None:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="480" '
        'style="border:1px solid rgba(128,128,128,.35);border-radius:8px;"></iframe>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def build_index(digest: str, file_name: str, _file_bytes: bytes, _api_key: str, _model_id: str):
    """Index a PDF. Cached on the file digest so re-runs never re-embed."""
    Settings.llm = NebiusLLM(model=_model_id, api_key=_api_key, temperature=0.1, max_tokens=4096)
    Settings.embed_model = NebiusEmbedding(model_name=EMBED_MODEL, api_key=_api_key)
    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, file_name)
        with open(path, "wb") as fh:
            fh.write(_file_bytes)
        docs = SimpleDirectoryReader(input_files=[path]).load_data()

    if not docs or not any(d.text.strip() for d in docs):
        raise ValueError("No extractable text found -- is this a scanned or image-only PDF?")
    return VectorStoreIndex.from_documents(docs, show_progress=False)


def reset_chat() -> None:
    st.session_state.messages = []
    gc.collect()


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
st.session_state.setdefault("messages", [])
st.session_state.setdefault("doc_digest", None)
st.session_state.setdefault("doc_name", None)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Setup")

    preset_key = read_secret("NEBIUS_API_KEY") or (os.getenv("NEBIUS_API_KEY") or "").strip()
    if preset_key:
        st.success("Nebius API key loaded from the environment.", icon="🔑")
    else:
        st.text_input(
            "Nebius API key",
            type="password",
            key="api_key",
            placeholder="Paste your key",
            help="Kept in your browser session only -- it is never stored server-side.",
        )
        st.caption("[Get a free Nebius API key](https://dub.sh/nebius)")

    model_label = st.selectbox("Model", list(MODELS.keys()), index=0)
    model_id = MODELS[model_label]

    st.divider()
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)

    file_bytes = b""
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        digest = hashlib.sha256(file_bytes).hexdigest()
        if digest != st.session_state.doc_digest:
            st.session_state.doc_digest = digest
            st.session_state.doc_name = uploaded.name
            reset_chat()
        st.caption(f"**{uploaded.name}** — {len(file_bytes) / 1024:.0f} KB")
        with st.expander("Preview", expanded=True):
            render_pdf(file_bytes)

    st.divider()
    st.button("Clear chat history", on_click=reset_chat, use_container_width=True)

# --------------------------------------------------------------------------- #
# Main pane
# --------------------------------------------------------------------------- #
st.title("🤖 Qwen3 RAG Chat")
st.caption(f"Retrieval-augmented chat over your PDFs, powered by {model_label} on Nebius AI")

api_key = resolve_api_key()

if not api_key:
    st.info("Add your Nebius API key in the sidebar to begin.", icon="🔑")
    st.stop()

if uploaded is None:
    st.info("Upload a PDF in the sidebar to start chatting with it.", icon="📄")
    st.stop()

try:
    with st.spinner("Indexing document -- embedding chunks with BAAI/bge-en-icl..."):
        index = build_index(
            st.session_state.doc_digest, uploaded.name, file_bytes, api_key, model_id
        )
except Exception as exc:  # shown to the user instead of a raw traceback
    st.error(f"Could not index this document: {exc}")
    st.stop()

query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
query_engine.update_prompts({"response_synthesizer:text_qa_template": QA_PROMPT})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("reasoning"):
            with st.expander("Model reasoning"):
                st.markdown(message["reasoning"])
        st.markdown(message["content"])

if prompt := st.chat_input(f"Ask anything about {st.session_state.doc_name}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        reasoning_box = st.empty()
        answer_box = st.empty()
        raw = ""
        try:
            response = query_engine.query(prompt)
            for token in response.response_gen:
                raw += token
                thoughts, answer = split_reasoning(raw)
                if thoughts:
                    with reasoning_box.container():
                        with st.expander("Model reasoning", expanded=False):
                            st.markdown(thoughts)
                answer_box.markdown(f"{answer} ▌" if answer else "_Thinking..._")
        except Exception as exc:
            answer_box.error(f"Request failed: {exc}")
            st.stop()

        thoughts, answer = split_reasoning(raw)
        answer_box.markdown(answer or "_No answer returned._")

        source_nodes = getattr(response, "source_nodes", None) or []
        if source_nodes:
            with st.expander("Retrieved sources"):
                for i, node in enumerate(source_nodes, start=1):
                    page = node.metadata.get("page_label", "?")
                    score = f"{node.score:.3f}" if node.score is not None else "n/a"
                    st.markdown(f"**{i}. Page {page}** — score `{score}`")
                    st.caption(node.get_content()[:600] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "reasoning": thoughts}
    )
