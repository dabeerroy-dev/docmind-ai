# ============================================
# FULL RAG SYSTEM - STAGE 2 + STAGE 3.1 + 3.2
# Multi Query + Hybrid + Reranking + Memory
# Multiple PDFs + Persistent Storage
# ============================================

# PyPDFLoader: reads PDF files page by page
from langchain_community.document_loaders import PyPDFLoader

# RecursiveCharacterTextSplitter: cuts text into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# BM25Okapi: keyword based search engine
from rank_bm25 import BM25Okapi

# chromadb: vector database
import chromadb

# Groq: fast FREE AI for clean answers
from groq import Groq

# os: file and folder operations
import os

# shutil: copy files from one place to another
import shutil

# ============================================
# SETUP — Initialize all tools
# ============================================

# Groq client for generating answers
# Replace with your real key for local testing
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ============================================
# PERSISTENT STORAGE SETUP
# DB_PATH: where ChromaDB saves data
# PDF_PATH: where uploaded PDFs are saved
# Both survive app restarts!
# ============================================

# Folder where vector database is stored
DB_PATH = "./database"

# Folder where PDF files are stored permanently
PDF_PATH = "./uploaded_pdfs"

# Create folders if they don't exist yet
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(PDF_PATH, exist_ok=True)

# PersistentClient saves to disk!
# Unlike EphemeralClient which loses data on restart
client = chromadb.PersistentClient(path=DB_PATH)

# ============================================
# FUNCTION 1: Load All Previously Saved PDFs
# When app restarts it finds old PDFs on disk
# and loads them automatically — persistence!
# ============================================
def load_saved_pdfs():

    # Get list of all .pdf files in saved folder
    saved = [
        os.path.join(PDF_PATH, f)
        for f in os.listdir(PDF_PATH)
        if f.endswith(".pdf")
    ]

    if saved:
        print(f"Found {len(saved)} previously saved PDFs!")
    else:
        print("No previously saved PDFs found!")

    return saved

# ============================================
# FUNCTION 2: Load Multiple PDFs
# Input: list of PDF file paths
# Output: list of text chunks + their metadata
# Each chunk has SOURCE info (which PDF, page)
# ============================================
def load_multiple_pdfs(pdf_paths):

    # Store all chunks from ALL pdfs here
    all_texts = []

    # Store metadata (source info) for each chunk
    all_metadatas = []

    # Get already indexed PDF names from database
    # So we don't index same PDF twice!
    collection = client.get_or_create_collection("pdf_docs")
    existing = collection.get()
    existing_sources = set()

    # Collect names of already indexed PDFs
    if existing["metadatas"]:
        for meta in existing["metadatas"]:
            existing_sources.add(meta["source"])

    print(f"Already indexed PDFs: {existing_sources}")

    # Loop through each PDF one by one
    for pdf_path in pdf_paths:

        # Get just filename from full path
        # "F:/RAG KR/doc.pdf" → "doc.pdf"
        filename = os.path.basename(pdf_path)

        # Skip if this PDF is already indexed!
        # No need to process same PDF twice
        if filename in existing_sources:
            print(f"Already indexed! Skipping: {filename}")
            continue

        print(f"Loading new PDF: {filename}")

        # Save PDF permanently to uploaded_pdfs folder
        # So it is available after app restarts!
        save_path = os.path.join(PDF_PATH, filename)
        if not os.path.exists(save_path):
            shutil.copy2(pdf_path, save_path)
            print(f"PDF saved permanently: {filename}")

        # Load PDF pages
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Split pages into smaller chunks
        # chunk_size=500: max 500 characters per chunk
        # chunk_overlap=50: chunks share 50 characters
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)

        # Process each chunk
        for chunk in chunks:

            # Skip empty or whitespace only chunks
            if not chunk.page_content.strip():
                continue

            # Add chunk text to list
            all_texts.append(chunk.page_content)

            # Add metadata: which PDF and page it came from
            all_metadatas.append({
                "source": filename,
                "page": chunk.metadata.get("page", 0)
            })

    print(f"New chunks to add: {len(all_texts)}")
    return all_texts, all_metadatas

# ============================================
# FUNCTION 3: Build/Update Vector Database
# Input: new texts + their metadata
# Output: updated ChromaDB collection
# Uses get_or_create so old data stays!
# ============================================
def build_database_with_metadata(texts, metadatas):

    # get_or_create = use existing OR make new
    # This keeps old chunks when adding new PDFs!
    collection = client.get_or_create_collection("pdf_docs")

    # Only add if there are new chunks to add
    if texts:
        print(f"Adding {len(texts)} new chunks to database...")

      collection.add(
    documents=texts,
    metadatas=metadatas,
    ids=[f"chunk_{existing_count + i}" 
         for i in range(len(texts))]
)
        print(f"Database now has {collection.count()} total chunks!")
    else:
        print(f"No new chunks. Database has {collection.count()} chunks!")

    return collection

# ============================================
# FUNCTION 4: Generate Query Variations
# Input: one question from user
# Output: 3-4 versions of same question
# WHY: Different words find different chunks!
# ============================================
def generate_queries(question):
    print("Generating query variations...")

    # Ask Groq to rephrase question 3 ways
    prompt = f"""Generate 3 different search queries
for this question to find information in a database.
Return ONLY 3 questions, one per line.
No numbering. No extra text.

Question: {question}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    # Split response by newline into list
    variations = (
        response.choices[0]
        .message.content
        .strip()
        .split("\n")
    )

    # Add original question to list too
    variations.append(question)

    # Remove any empty lines
    variations = [q for q in variations if q.strip()]

    print(f"Generated {len(variations)} query versions!")
    return variations

# ============================================
# FUNCTION 5: BM25 Keyword Search
# Input: question + all text chunks
# Output: top chunks by keyword matching
# WHY: finds exact technical terms perfectly!
# ============================================
def bm25_search(question, all_texts, top_k=3):

    # Split each chunk into individual words
    # BM25 counts how often words appear
    tokenized_chunks = [
        text.lower().split()
        for text in all_texts
    ]

    # Build BM25 search index from all chunks
    bm25 = BM25Okapi(tokenized_chunks)

    # Split question into words too
    tokenized_question = question.lower().split()

    # Score each chunk: how many question words appear?
    scores = bm25.get_scores(tokenized_question)

    # Get indices of top_k highest scoring chunks
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    # Return top matching chunk texts
    return [all_texts[i] for i in top_indices]

# ============================================
# FUNCTION 6: Hybrid Search with Sources
# Input: question + collection + all texts
# Output: best chunks + their source info
# Combines Vector Search + BM25 together!
# ============================================
def search_with_sources(question, collection, all_texts):

    # Step 1: Generate multiple query versions
    queries = generate_queries(question)

    # Store all found chunks and their sources
    all_chunks = []
    all_sources = []

    # Step 2: Search with each query version
    for query in queries:

        # --- Vector Search ---
        # Finds chunks by semantic meaning
        vector_results = collection.query(
    query_texts=[query],
    n_results=3,
    include=["documents", "metadatas"]
)

        # Add vector results with their sources
        for doc, meta in zip(
            vector_results["documents"][0],
            vector_results["metadatas"][0]
        ):
            all_chunks.append(doc)
            all_sources.append(
                f"{meta['source']} (Page {meta['page']+1})"
            )

        # --- BM25 Search ---
        # Finds chunks by exact keyword matching
        bm25_chunks = bm25_search(query, all_texts, top_k=3)
        for chunk in bm25_chunks:
            all_chunks.append(chunk)
            # BM25 doesn't return metadata
            # so we use "Keyword Match" as source
            all_sources.append("Keyword Match")

    # Step 3: Remove duplicate chunks
    seen = set()
    unique_chunks = []
    unique_sources = []

    for chunk, source in zip(all_chunks, all_sources):
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)
            unique_sources.append(source)

    # Step 4: Rerank by relevance score
    if unique_chunks:
        # Create [question, chunk] pairs for reranker
        # Simple keyword reranking - no torch needed!
q_words = set(question.lower().split())
scored = []
for chunk, source in zip(unique_chunks, unique_sources):
    c_words = set(chunk.lower().split())
    score = len(q_words & c_words) / max(len(q_words | c_words), 1)
    scored.append((score, chunk, source))
ranked = sorted(scored, reverse=True)

        # Take top 3 most relevant chunks
        best_chunks = [c for _, c, _ in ranked[:3]]
        best_sources = [s for _, _, s in ranked[:3]]
    else:
        best_chunks = unique_chunks[:3]
        best_sources = unique_sources[:3]

    print(f"Found {len(best_chunks)} best chunks!")
    return best_chunks, best_sources

# ============================================
# FUNCTION 7: Generate Final Answer
# Input: question + best chunks + sources + history
# Output: clean professional answer with source
# ============================================
def generate_answer_with_sources(
    question, chunks, sources, chat_history
):
    print("Generating answer...")

    # Join best chunks into one context paragraph
    context = "\n\n".join(chunks)

    # Build conversation history for memory
    # Only last 4 messages to keep prompt short
    history_text = ""
    for msg in chat_history[-4:]:
        role = msg["role"].upper()
        history_text += f"{role}: {msg['content']}\n"

    # Final prompt with context + history + sources
    prompt = f"""You are a professional AI assistant.
Answer using ONLY the context below.
At the end mention which document the answer came from.
If not in context say 'I dont know'.

Previous conversation:
{history_text}

Context:
{context}

Sources: {', '.join(set(sources))}

Question: {question}
Answer:"""

    # Send to Groq for clean answer
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ============================================
# MAIN PIPELINE
# Combines ALL functions together!
# ============================================
def ask_rag(question, collection, all_texts, chat_history):

    # Search with hybrid + source tracking
    best_chunks, sources = search_with_sources(
        question, collection, all_texts
    )

    # Generate clean answer with sources
    answer = generate_answer_with_sources(
        question,
        best_chunks,
        sources,
        chat_history
    )

    return answer, sources

# ============================================
# RUN THE SYSTEM!
# ============================================
if __name__ == "__main__":

    print("=== STARTING PERSISTENT RAG SYSTEM ===")

    # Step 1: Load previously saved PDFs from disk
    # This is persistence — remembers old PDFs!
    saved_pdfs = load_saved_pdfs()

    # Step 2: Add new PDFs you want to index
    new_pdfs = ["DB DNL.pdf"]

    # Step 3: Combine saved + new PDFs
    # set() removes duplicates automatically
    all_pdf_paths = list(set(saved_pdfs + new_pdfs))
    print(f"Total PDFs to process: {len(all_pdf_paths)}")

    # Step 4: Load and chunk all PDFs
    texts, metadatas = load_multiple_pdfs(all_pdf_paths)

    # Step 5: Build/update vector database
    collection = build_database_with_metadata(
        texts, metadatas
    )

    # Step 6: Get ALL texts for BM25 search
    # BM25 needs all texts not just new ones
    all_data = collection.get()
    all_texts = all_data["documents"]

    # Empty chat history for this session
    chat_history = []

    print("\n=== PERSISTENT RAG READY! ===")
    print(f"Total chunks in database: {collection.count()}")
    print("Data is saved! Survives restarts!")
    print("Type 'quit' to exit\n")

    # Chat loop
    while True:
        question = input("You: ")

        # Exit if user types quit
        if question.lower() == "quit":
            print("Goodbye! Data is saved!")
            break

        # Get answer with sources
        answer, sources = ask_rag(
            question,
            collection,
            all_texts,
            chat_history
        )

        # Save to chat memory
        chat_history.append({
            "role": "user",
            "content": question
        })
        chat_history.append({
            "role": "assistant",
            "content": answer
        })

        # Show answer and sources
        print(f"\nAI: {answer}")
        print(f"\nSources: {set(sources)}")
        print("-" * 40)
