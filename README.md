# RAG Policy Search API

A FastAPI-based Retrieval-Augmented Generation (RAG) system for searching HR policies.  
This service allows you to upload a PDF or text policy document, automatically chunk + embed it, and then query the document using a Groq LLM with context-aware responses.

## Features
- PDF & text ingestion  
- Automatic text chunking  
- SentenceTransformer embeddings  
- Cosine similarity-based vector search  
- Context-aware LLM answering using Groq API  
- Fully containerized with Docker & Docker Compose  
- REST endpoints to upload policy & query it

## 📂 Project Structure
rag-five/
│── Dockerfile
│── docker-compose.yml
│── req.txt
│── README.md
│── .env (optional)
└── app/
    └── main.py

## ⚙️ Architecture Overview

### 1. Document Ingestion
- User uploads a PDF or `.txt` file  
- PDF text extracted using PyPDF2  
- Text is split into overlapping chunks  

### 2. Embedding Generation
- Embedding model used: `all-MiniLM-L6-v2`  
- Stored in memory:  
  - `CHUNKS[]` → raw text chunks  
  - `EMBEDS[]` → vector embeddings  

### 3. Semantic Search
- Cosine similarity is used to find the top-k similar chunks  
- These top chunks are passed as context to the LLM  

### 4. LLM Answer (Groq)
- Uses ChatGroq **Llama 3.3-70B** model  
- Custom HR-policy prompt ensures answers come **only from the uploaded document**

## 🧪 Test Locally (Without Docker)
1.Clone the project
- git clone <your-repo-url>
- cd app
  
2.Install dependencies
- pip install -r requirements.txt

Or use a virtual environment:

- python3 -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt

3.Set environment variable
- export GROQ_API_KEY="your_groq_api_key"

Or create a .env file:
- GROQ_API_KEY=your_groq_api_key_here

4️.Run FastAPI
- uvicorn main:app --reload

5.Open Swagger UI
- http://127.0.0.1:8000/docs

## 📡 API Endpoints
### 1.POST /upload-policy

Upload a PDF or text file.

Example Response
{
  "message": "Policy uploaded successfully",
  "chunks": 25
}

### 2.POST /query

Ask a question about the uploaded policy.

Body
{
  "question": "What is the leave policy?"
}

Example Response
{
  "answer": "The leave policy states...",
  "source_context": "text from top relevant chunks..."
}

## 🐳 Running with Docker

### 1. Build and run the container (first time only)

This builds the image and starts the API:

- docker compose up --build

### 2. Start the API again later (no rebuild needed)
-docker compose up

Now open:
-http://localhost:8000/docs






