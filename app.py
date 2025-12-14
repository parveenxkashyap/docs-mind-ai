import streamlit as st
import os
from dotenv import load_dotenv
import asyncio
from datetime import datetime
import time
import tempfile

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
RETRIEVER_K = 4
DEFAULT_SYSTEM_MESSAGE = """
You are Document RAG Assistant 📄🤖. 
Your role is to help users understand and explore the content of uploaded documents.
"""

load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def configure_page():
    st.set_page_config(
        page_title="DocMind",
        page_icon="📄",
        layout="centered",
    )
    st.title("📄DocMind")
    st.markdown("### Chat with your documents using AI")


def center_app():
    st.markdown(
        """
        <style>
        .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            max-width: 800px;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def handle_sidebar():
    st.sidebar.header("🔑 Configuration")

    api_key = st.sidebar.text_input(
        "Your Google Gemini API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
    )

    if api_key:
        st.session_state.api_key = api_key
        os.environ["GOOGLE_API_KEY"] = api_key

    selected_model = st.sidebar.selectbox(
        "Generation Models",
        ["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
    )

    st.session_state.model = selected_model

    uploaded_file = st.file_uploader(
        "📁 Upload your document",
        type=["pdf", "txt"],
    )

    return selected_model, uploaded_file, api_key


init_session_state()
configure_page()
center_app()
selected_model, uploaded_file, user_api_key = handle_sidebar()
