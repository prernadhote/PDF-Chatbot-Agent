# PDF Chatbot Agent using Generative AI

## Project Overview

This project is an AI-powered PDF Chatbot that allows
users to upload PDF documents and ask questions about
their content.

The application uses Retrieval-Augmented Generation
(RAG) to retrieve relevant information from uploaded
PDF documents and generate answers using an LLM.

## Technologies Used

- Python
- Streamlit
- LangChain
- Ollama
- Llama 3.2
- Nomic Embeddings
- FAISS Vector Database
- PyPDF

## Project Architecture

1. Upload PDF document.
2. Extract text using PyPDFLoader.
3. Split text into smaller chunks.
4. Generate embeddings.
5. Store embeddings in FAISS.
6. Retrieve relevant chunks based on user questions.
7. Generate answers using Llama 3.2.

## Installation

Clone the repository:

git clone https://github.com/prernadhote/PDF-Chatbot-Agent.git

Navigate to the project:

cd PDF-Chatbot-Agent

Install dependencies:

pip install -r requirements.txt

Install Ollama models:

ollama pull llama3.2
ollama pull nomic-embed-text

## Run Application

streamlit run app.py

## Features

- PDF document upload
- PDF text extraction
- Text chunking
- Semantic search
- AI-powered question answering
- Source page references
- Interactive chat interface

## Author

Prerna Dhote