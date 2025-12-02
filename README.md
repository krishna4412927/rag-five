A FastAPI-based Retrieval-Augmented Generation (RAG) system for searching HR policies.
This service allows you to upload a PDF or text policy document, automatically chunk + embed it, and then query the document using a Groq LLM with context-aware responses.

## Features
-PDF & text ingestion
-Automatic text chunking
-SentenceTransformer embeddings
-Cosine similarity-based vector search
-Context-aware LLM answering using Groq API
-Fully containerized with Docker & Docker Compose
-REST endpoints to upload policy & query it

## Project Structure




⚙️ Architecture Overview
1. Document Ingestion
-User uploads a PDF or .txt file
-PDF text extracted using PyPDF2
-Text is split into overlapping chunks using a custom chunker

2. Embedding Generation

Each chunk is encoded using:

SentenceTransformer – all-MiniLM-L6-v2

Stored in memory:

CHUNKS[] → raw text chunks

EMBEDS[] → vector embeddings

3. Semantic Search

Cosine similarity is used to find the top-k similar chunks

These chunks are passed as context to the LLM

4. LLM Answer (Groq)

Uses ChatGroq Llama 3.3-70B model

Custom HR-policy prompt ensures answers come only from the document
