# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import asyncio
from datetime import datetime
import time
import tempfile

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
RETRIEVER_K = 4
DEFAULT_SYSTEM_MESSAGE = """
You are Document RAG Assistant 📄🤖. 
Your role is to help users understand and explore the content of uploaded documents.

Follow these rules:
1. Always prioritize the document context when answering questions.
2. If the answer is not in the document, clearly say you don't know.
3. Keep responses friendly, clear, and concise.
"""

# Load environment variables
load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


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


init_session_state()
configure_page()
center_app()
