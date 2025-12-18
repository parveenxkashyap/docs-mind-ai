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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader


def handle_document_processing(uploaded_file=""):
    if st.button("🚀 Process Document"):
        if not uploaded_file:
            st.error("Please upload a document first!")
            return

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = TextLoader(tmp_file_path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_documents(chunks, embeddings)

        st.session_state["retriever"] = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": RETRIEVER_K}
        )

        st.success("Document processed successfully!")


if uploaded_file:
    handle_document_processing(uploaded_file)
