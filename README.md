# EffiHire RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for EffiHire that answers questions based on company documentation.

## Features

- Semantic search using sentence embeddings
- Conversation memory with PostgreSQL backend
- PDF document processing
- Support ticket system

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (see below)
4. Run the server: `python server.py`

## Environment Variables

Create a `.env` file with:
FLASK_SECRET_KEY=your_secret_key
DB_NAME=effihire DB_USER=your_db_user
DB_PASSWORD=your_db_password DB_HOST=your_db_host
DB_PORT=5432 GEMINI_API_KEY=your_gemini_api_key
PDF_PATH=./EffiHire_document.pdf

## Deployment

This project can be deployed to:
- Google Cloud Run
- Render
- Any platform supporting Python/Flask applications

