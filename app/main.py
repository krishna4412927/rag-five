
import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "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")

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"  # change to "cuda" if you have GPU
)

chat_client = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)

app = FastAPI(title="RAG Policy Search API")

# In-memory storage
CHUNKS = []
EMBEDS = []

def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, size=1500, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def embed(text: str):
    """Use HuggingFace Transformer model for embeddings"""
    return embed_model.encode(text)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query: str, k=3):
    q_emb = embed(query)
    scores = [(cosine_sim(q_emb, emb), i) for i, emb in enumerate(EMBEDS)]
    scores.sort(reverse=True)
    top = scores[:k]
    return [CHUNKS[i] for _, i in top]

 
def generate_answer(question: str, context_docs):
    """
    Generate an answer using ChatGroq LLM properly.
    """
    # Combine all context chunks into a single string
    context_text = "\n---\n".join(context_docs)

    # HR policy prompt
    hr_policy_prompt = f"""
You are an HR assistant.
Answer ONLY using the provided policy text.
If the answer is not found in the policy, reply: "I cannot find this in the policy."

Context:
{context_text}

Question:
{question}

Answer:
"""

    # Initialize Groq model (use the global one or create new)
    grok_model = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),  # Use env var properly
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    # CORRECT WAY: Use .invoke() method
    try:
        response = grok_model.invoke(hr_policy_prompt)
        return response.content  # Extract the content from response
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# Pydantic model
# -------------------------
class Query(BaseModel):
    question: str

# -------------------------
# FastAPI Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "RAG Policy Search API running. Use /docs to test."}

@app.post("/upload-policy")
async def upload_policy(file: UploadFile = File(...)):
    global CHUNKS, EMBEDS
    CHUNKS = []
    EMBEDS = []

    if file.filename.endswith(".pdf"):
        text = extract_pdf_text(file.file)
    else:
        text = (await file.read()).decode("utf-8")

    chunks = chunk_text(text)

    for c in chunks:
        emb = embed(c)
        CHUNKS.append(c)
        EMBEDS.append(emb)

    return {"message": "Policy uploaded successfully", "chunks": len(chunks)}

@app.post("/query")
async def query(q: Query):
    if not CHUNKS:
        return {"answer": None, "source_context": "No policy uploaded yet."}

    top_chunks = search(q.question, k=10)
    context_docs = top_chunks  # Pass the chunks as context
    answer = generate_answer(q.question, context_docs)

    return {"answer": answer, "source_context": "\n---\n".join(top_chunks)}
