# ============================================
# DOCMIND AI — FastAPI Backend
# REST API for RAG System
# Any app can connect to this!
# ============================================

# FastAPI: main framework for building APIs
from fastapi import FastAPI, UploadFile, File

# HTTPException: return error messages properly
from fastapi.exceptions import HTTPException

# BaseModel: define request/response structure
from pydantic import BaseModel

# All our RAG functions from full_rag.py
from full_rag import (
    load_multiple_pdfs,
    build_database_with_metadata,
    ask_rag,
    client,
    embedder,
    PDF_PATH,
    DB_PATH
)

# os: file operations
import os

# shutil: save uploaded files
import shutil

# chromadb: for database operations
import chromadb

# ============================================
# CREATE FASTAPI APP
# This is the main application object
# All endpoints are added to this!
# ============================================
app = FastAPI(
    title="DocMind AI API",
    description="RAG-powered document intelligence API",
    version="1.0.0"
)

# ============================================
# CHAT HISTORY STORAGE
# Stores conversation per session
# In production use Redis or database!
# ============================================
chat_histories = {}

# ============================================
# REQUEST MODELS
# Define what data API expects to receive
# BaseModel = automatic validation!
# ============================================

# Model for asking questions
class QuestionRequest(BaseModel):
    # question: the user's question
    question: str
    # session_id: tracks conversation memory
    # default is "default" session
    session_id: str = "default"

# ============================================
# RESPONSE MODELS
# Define what data API sends back
# ============================================

# Model for question answers
class AnswerResponse(BaseModel):
    # answer: the AI's answer
    answer: str
    # sources: which PDFs were used
    sources: list
    # session_id: which session answered
    session_id: str
    
    # ============================================
# ENDPOINT 1: Health Check
# URL: GET /
# Purpose: Check if API is running
# Like knocking door to see if someone home!
# ============================================
@app.get("/")
def health_check():
    # Return simple status message
    return {
        "status": "running",
        "message": "DocMind AI API is live!",
        "version": "1.0.0"
    }

# ============================================
# ENDPOINT 2: Upload PDF
# URL: POST /upload
# Purpose: Upload PDF and index it
# Client sends PDF → we process and store!
# ============================================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # Check if uploaded file is a PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files allowed!"
        )

    # Save uploaded file to pdf folder
    save_path = os.path.join(PDF_PATH, file.filename)

    # Write file to disk
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"PDF uploaded: {file.filename}")

    # Index the new PDF
    texts, metadatas = load_multiple_pdfs([save_path])
    build_database_with_metadata(texts, metadatas)

    # Return success message
    return {
        "status": "success",
        "filename": file.filename,
        "message": f"{file.filename} uploaded and indexed!"
    }

# ============================================
# ENDPOINT 3: Ask Question
# URL: POST /ask
# Purpose: Ask question and get AI answer
# This is the MAIN endpoint!
# ============================================
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):

    # Check if database has any PDFs
    collection = client.get_or_create_collection("pdf_docs")

    if collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No PDFs uploaded yet! Upload a PDF first."
        )

    # Get or create chat history for this session
    # Each session_id has its own memory!
    if request.session_id not in chat_histories:
        chat_histories[request.session_id] = []

    # Get this session's chat history
    history = chat_histories[request.session_id]

    # Get all texts for BM25 search
    all_data = collection.get()
    all_texts = all_data["documents"]

    # Get answer from RAG pipeline
    answer, sources = ask_rag(
        request.question,
        collection,
        all_texts,
        history
    )

    # Save to chat history (memory!)
    history.append({
        "role": "user",
        "content": request.question
    })
    history.append({
        "role": "assistant",
        "content": answer
    })

    # Return answer with sources
    return AnswerResponse(
        answer=answer,
        sources=list(set(sources)),
        session_id=request.session_id
    )

# ============================================
# ENDPOINT 4: List All PDFs
# URL: GET /pdfs
# Purpose: See all uploaded PDFs
# ============================================
@app.get("/pdfs")
def list_pdfs():

    # Get all PDF files from folder
    pdfs = [
        f for f in os.listdir(PDF_PATH)
        if f.endswith(".pdf")
    ]

    # Get database stats
    collection = client.get_or_create_collection("pdf_docs")

    return {
        "total_pdfs": len(pdfs),
        "pdfs": pdfs,
        "total_chunks": collection.count()
    }

# ============================================
# ENDPOINT 5: Clear Chat History
# URL: DELETE /history/{session_id}
# Purpose: Reset conversation memory
# ============================================
@app.delete("/history/{session_id}")
def clear_history(session_id: str):

    # Delete this session's history
    if session_id in chat_histories:
        chat_histories[session_id] = []
        return {
            "status": "success",
            "message": f"History cleared for {session_id}"
        }
    else:
        return {
            "status": "not found",
            "message": f"No history for {session_id}"
        }

# ============================================
# ENDPOINT 6: Reset Database
# URL: DELETE /reset
# Purpose: Clear all PDFs and database
# Use carefully! Deletes everything!
# ============================================
@app.delete("/reset")
def reset_database():

    # Delete all PDF files
    for f in os.listdir(PDF_PATH):
        if f.endswith(".pdf"):
            os.remove(os.path.join(PDF_PATH, f))

    # Delete database collection
    try:
        client.delete_collection("pdf_docs")
    except:
        pass

    # Clear all chat histories
    chat_histories.clear()

    return {
        "status": "success",
        "message": "Database reset! All data deleted!"
    }

# ============================================
# RUN THE API SERVER
# ============================================
if __name__ == "__main__":
    import uvicorn

    # uvicorn: runs the FastAPI server
    # host="0.0.0.0": accessible from anywhere
    # port=8000: runs on port 8000
    # reload=True: auto-restarts on code change
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )