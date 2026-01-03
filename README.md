# 📄 DocsMind AI

> Transform any document into an interactive AI conversation using RAG (Retrieval Augmented Generation)

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.32+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Last Commit](https://img.shields.io/github/last-commit/parveenxkashyap/docs-mind-ai)
[![Live Demo](https://img.shields.io/badge/demo-online-green.svg)](http)

## 🎯 Overview

DocsMind AI is a Streamlit web application that enables you to have intelligent conversations with your document content. Simply upload a PDF or text file, and the app will process the document to create a searchable knowledge base that you can query using natural language.


## 🌐 Live Demo

Try it out here: [DocsMind AI Live Demo](https)

## ✨ Features

- 📁 **Multi-format Support**: Process PDF and text files seamlessly
- 🤖 **Multiple AI Models**: Support for various Google Gemini models (2.5 Pro, Flash, 2.0 Flash, etc.)
- 💬 **Interactive Chat**: Natural language conversation with document content
- 🔍 **Smart Search**: Vector-based similarity search using FAISS
- 📊 **Session Management**: Chat history, export functionality, and session persistence
- 🎨 **Modern UI**: Clean, responsive Streamlit interface with real-time updates
- 📈 **Progress Tracking**: Visual feedback during document processing
- 🔄 **Streaming Responses**: Real-time AI response streaming with typing indicators
- 🛡️ **Fallback System**: Automatic HuggingFace embeddings if Google quota exceeded

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI/ML**: Google Gemini API, LangChain
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Google Generative AI Embeddings
- **Document Processing**: PyPDFLoader, TextLoader
- **Environment**: Python 3.11+, Docker support

## 📋 Prerequisites

- Python 3.11 or higher
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))
- Internet connection for API access
