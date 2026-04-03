# ============================================
# DOCMIND AI — International Level SaaS
# Firebase Auth + Hybrid RAG + Beautiful UI
# FIXED: ChromaDB embedding issue + deployment errors
# ============================================
import streamlit as st
from auth import signup_user, login_user
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from pypdf import PdfReader         # FIXED: use pypdf directly (no langchain loader needed)
from fastembed import TextEmbedding  # FIXED: lightweight embeddings, no torch!
from groq import Groq
import tempfile
import os
import numpy as np

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="DocMind AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# WORLD CLASS CSS
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

.stApp {
    background: linear-gradient(135deg,
        #0a0a1a 0%, #0d1b2a 40%,
        #0a1628 70%, #0d0a1a 100%);
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: rgba(10,10,30,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}

#MainMenu, footer, header { visibility: hidden; }

.main .block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
}

.brand-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

.brand-sub {
    text-align: center;
    color: rgba(255,255,255,0.4);
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.user-bubble {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
}

.user-bubble-inner {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 0.9rem;
    line-height: 1.5;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3);
}

.ai-bubble {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
    gap: 10px;
    align-items: flex-start;
}

.ai-bubble-inner {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.9);
    padding: 12px 18px;
    border-radius: 4px 18px 18px 18px;
    max-width: 70%;
    font-size: 0.9rem;
    line-height: 1.5;
}

.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}

.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stButton button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: white !important;
}

hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================
# EMBEDDING MODEL — loaded once and cached
# Using fastembed: lightweight, no PyTorch needed!
# @st.cache_resource means it loads only ONCE
# even when the user interacts with the app
# ============================================
@st.cache_resource
def load_embedding_model():
    """
    Load the fastembed model once and reuse it.
    This prevents reloading the model on every interaction.
    ~50MB download, much faster than torch's 2GB!
    """
    print("Loading fastembed model...")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print("Embedding model ready!")
    return model


def embed_texts(model, texts: list) -> np.ndarray:
    """
    Convert a list of text strings into vectors (embeddings).
    Returns a numpy array of shape (num_texts, embedding_dim).
    """
    # fastembed returns a generator → convert to list → stack into array
    embeddings = list(model.embed(texts))
    return np.array(embeddings)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculate how similar two vectors are.
    Returns a number between 0 (different) and 1 (identical).
    This is how we do semantic search without ChromaDB's query_texts!
    """
    # Dot product divided by product of magnitudes
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def vector_search(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    texts: list,
    top_k: int = 3
) -> list:
    """
    Search for the most similar text chunks to a query.
    Pure numpy — no ChromaDB query_texts needed!

    How it works:
    1. Calculate similarity between query and EVERY chunk
    2. Sort by similarity score
    3. Return top_k best matches
    """
    # Calculate similarity to every stored chunk
    scores = [
        cosine_similarity(query_embedding, doc_embeddings[i])
        for i in range(len(doc_embeddings))
    ]
    # Get indices of top_k highest scores
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    # Return the actual text chunks
    return [texts[i] for i in top_indices]


# ============================================
# SESSION STATE INITIALIZATION
# ============================================
for key, val in {
    "logged_in": False,
    "token": None,
    "username": "",
    "email": "",
    "uid": "",
    "messages": [],
    "pdf_ready": False,
    "doc_embeddings": None,   # FIXED: store embeddings as numpy array
    "all_texts": [],
    "pdf_name": "",
    "auth_mode": "login"
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================
# AUTH PAGE
# ============================================
if not st.session_state.logged_in:

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='brand-title'>⚡ DocMind AI</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='brand-sub'>Intelligent Document Platform</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:

        # Toggle tabs
        t1, t2 = st.columns(2)
        with t1:
            if st.button(
                "Sign In",
                use_container_width=True,
                type="primary" if st.session_state.auth_mode == "login"
                else "secondary"
            ):
                st.session_state.auth_mode = "login"
                st.rerun()
        with t2:
            if st.button(
                "Sign Up",
                use_container_width=True,
                type="primary" if st.session_state.auth_mode == "signup"
                else "secondary"
            ):
                st.session_state.auth_mode = "signup"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- LOGIN ----
        if st.session_state.auth_mode == "login":
            st.markdown("""
            <div style='color:white; font-family:Syne,sans-serif;
                        font-size:1.4rem; font-weight:800;
                        margin-bottom:1rem;'>
                Welcome Back 👋
            </div>""", unsafe_allow_html=True)

            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input(
                "Password", type="password",
                placeholder="Enter your password"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Sign In →", use_container_width=True):
                if email and password:
                    with st.spinner("Signing in..."):
                        result = login_user(email, password)
                    if result["success"]:
                        st.session_state.logged_in = True
                        st.session_state.token = result["token"]
                        st.session_state.email = result["email"]
                        st.session_state.username = result["name"]
                        st.session_state.uid = result["uid"]
                        st.rerun()
                    else:
                        st.error("Wrong email or password!")
                else:
                    st.warning("Please fill all fields!")

        # ---- SIGNUP ----
        else:
            st.markdown("""
            <div style='color:white; font-family:Syne,sans-serif;
                        font-size:1.4rem; font-weight:800;
                        margin-bottom:1rem;'>
                Create Account 🚀
            </div>""", unsafe_allow_html=True)

            name = st.text_input("Full Name", placeholder="Your name")
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input(
                "Password", type="password",
                placeholder="Min 6 characters"
            )
            confirm = st.text_input(
                "Confirm Password", type="password",
                placeholder="Repeat password"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Create Account →", use_container_width=True):
                if not all([name, email, password, confirm]):
                    st.warning("Please fill all fields!")
                elif len(password) < 6:
                    st.error("Password must be 6+ characters!")
                elif password != confirm:
                    st.error("Passwords do not match!")
                else:
                    with st.spinner("Creating account..."):
                        result = signup_user(email, password, name)
                    if result["success"]:
                        st.success("Account created! Please sign in!")
                        st.session_state.auth_mode = "login"
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")

# ============================================
# MAIN DASHBOARD
# ============================================
else:
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:1.5rem 1rem; border-bottom:
                    1px solid rgba(99,102,241,0.2);'>
            <div style='font-family:Syne,sans-serif;
                        font-size:1.4rem; font-weight:800;
                        background:linear-gradient(135deg,#6366f1,#06b6d4);
                        -webkit-background-clip:text;
                        -webkit-text-fill-color:transparent;'>
                ⚡ DocMind AI
            </div>
        </div>
        <div style='padding:1rem; margin-top:0.5rem;'>
            <div style='display:flex; align-items:center; gap:10px;'>
                <div style='width:38px; height:38px;
                            background:linear-gradient(135deg,#6366f1,#06b6d4);
                            border-radius:50%; display:flex;
                            align-items:center; justify-content:center;
                            font-size:1rem;'>
                    👤
                </div>
                <div>
                    <div style='color:white; font-weight:600; font-size:0.9rem;'>
                        {st.session_state.username}
                    </div>
                    <div style='color:rgba(255,255,255,0.4); font-size:0.75rem;'>
                        {st.session_state.email}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "",
            ["🏠  Dashboard", "💬  Chat", "📊  Analytics", "⚙️  Settings"],
            label_visibility="collapsed"
        )

        st.markdown("<br>" * 5, unsafe_allow_html=True)

        if st.button("🚪 Sign Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ---- DASHBOARD ----
    if "Dashboard" in page:
        st.markdown(f"""
        <div style='margin-bottom:2rem;'>
            <div style='font-family:Syne,sans-serif;
                        font-size:2rem; font-weight:800; color:white;'>
                Welcome, {st.session_state.username}! 👋
            </div>
            <div style='color:rgba(255,255,255,0.4); margin-top:4px;'>
                Your AI document intelligence platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, num, label in [
            (c1, "⚡", "Hybrid Search"),
            (c2, "4x", "Query Expansion"),
            (c3, "83%", "Accuracy"),
            (c4, "🔒", "Secured")
        ]:
            with col:
                st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-number'>{num}</div>
                    <div style='color:rgba(255,255,255,0.5);
                                font-size:0.8rem; margin-top:4px;
                                text-transform:uppercase; letter-spacing:1px;'>
                        {label}
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 Go to **Chat** to upload PDF and ask questions!")

    # ---- CHAT ----
    elif "Chat" in page:
        st.markdown("""
        <div style='margin-bottom:1.5rem;'>
            <div style='font-family:Syne,sans-serif;
                        font-size:1.8rem; font-weight:800; color:white;'>
                💬 AI Chat
            </div>
            <div style='color:rgba(255,255,255,0.4);'>
                Upload PDF and chat instantly
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Load embedding model (cached — only loads once)
        embed_model = load_embedding_model()

        # ---- PDF UPLOAD ----
        if not st.session_state.pdf_ready:
            uploaded = st.file_uploader(
                "Upload PDF", type="pdf",
                label_visibility="collapsed"
            )

            if uploaded:
                with st.spinner("⚡ Processing your PDF..."):

                    # STEP 1: Extract text from PDF using pypdf directly
                    # (removed PyPDFLoader which needs langchain_community)
                    reader = PdfReader(uploaded)
                    raw_text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            raw_text += page_text + "\n"

                    # STEP 2: Split into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    chunks = splitter.split_text(raw_text)
                    # Remove empty chunks
                    texts = [c.strip() for c in chunks if c.strip()]

                    # STEP 3: Embed all chunks using fastembed
                    # This creates a numpy array of vectors
                    # Shape: (number_of_chunks, 384)
                    # We store these in session_state for reuse
                    doc_embeddings = embed_texts(embed_model, texts)

                    # STEP 4: Save to session state
                    st.session_state.doc_embeddings = doc_embeddings
                    st.session_state.all_texts = texts
                    st.session_state.pdf_ready = True
                    st.session_state.pdf_name = uploaded.name

                st.rerun()

        # ---- CHAT INTERFACE ----
        else:
            st.success(f"✅ {st.session_state.pdf_name} ready!")

            if st.button("📄 Upload New PDF"):
                # Reset PDF state but keep user logged in
                st.session_state.pdf_ready = False
                st.session_state.doc_embeddings = None
                st.session_state.all_texts = []
                st.session_state.messages = []
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # Show chat history
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-bubble">'
                        f'<div class="user-bubble-inner">'
                        f'{msg["content"]}</div></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="ai-bubble">'
                        f'<div style="width:32px;height:32px;'
                        f'background:linear-gradient(135deg,#06b6d4,#6366f1);'
                        f'border-radius:50%;display:flex;align-items:center;'
                        f'justify-content:center;font-size:0.8rem;'
                        f'flex-shrink:0;">⚡</div>'
                        f'<div class="ai-bubble-inner">'
                        f'{msg["content"]}</div></div>',
                        unsafe_allow_html=True
                    )

            # Chat input box
            question = st.chat_input("Ask anything about your document...")

            if question:
                groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

                with st.spinner("⚡ Thinking..."):

                    # ---- STEP 1: MULTI-QUERY GENERATION ----
                    # Ask the LLM to rephrase the question 3 ways
                    # More query variations = better retrieval!
                    r = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{
                            "role": "user",
                            "content": (
                                f"Generate 3 search queries for: {question}\n"
                                f"Return 3 questions only, one per line."
                            )
                        }],
                        max_tokens=200,
                        temperature=0.7
                    )
                    # Split response into individual queries
                    generated = r.choices[0].message.content.strip().split("\n")
                    # Clean up numbered lists like "1. What is..."
                    generated = [
                        q.lstrip("0123456789.-) ").strip()
                        for q in generated if q.strip()
                    ]
                    # Always include the original question too
                    queries = generated + [question]

                    # ---- STEP 2: HYBRID SEARCH ----
                    # For each query variation, do BOTH:
                    # A) Vector search (semantic meaning)
                    # B) BM25 keyword search (exact words)
                    all_chunks = []

                    for q in queries:

                        # A) VECTOR SEARCH — embed query and find similar chunks
                        # This finds chunks with similar MEANING
                        q_embedding = embed_texts(embed_model, [q])[0]
                        vector_results = vector_search(
                            q_embedding,
                            st.session_state.doc_embeddings,
                            st.session_state.all_texts,
                            top_k=3
                        )
                        all_chunks.extend(vector_results)

                        # B) BM25 KEYWORD SEARCH — find chunks with matching words
                        # This catches exact terminology vector search might miss
                        if st.session_state.all_texts:
                            # Tokenize all chunks (split into words)
                            tokenized = [
                                t.lower().split()
                                for t in st.session_state.all_texts
                            ]
                            bm25 = BM25Okapi(tokenized)
                            # Score each chunk for this query
                            scores = bm25.get_scores(q.lower().split())
                            # Get top 3 scoring chunks
                            top_idx = sorted(
                                range(len(scores)),
                                key=lambda i: scores[i],
                                reverse=True
                            )[:3]
                            all_chunks.extend([
                                st.session_state.all_texts[i]
                                for i in top_idx
                            ])

                    # ---- STEP 3: DEDUPLICATE ----
                    # Remove duplicate chunks (same text found by both searches)
                    unique_chunks = list(dict.fromkeys(all_chunks))

                    # ---- STEP 4: RERANK ----
                    # Score each chunk by word overlap with the question
                    # Best chunks rise to the top
                    q_words = set(question.lower().split())
                    scored = []
                    for chunk in unique_chunks:
                        c_words = set(chunk.lower().split())
                        # Jaccard similarity: shared words / all words
                        overlap = len(q_words & c_words)
                        union = len(q_words | c_words)
                        score = overlap / max(union, 1)
                        scored.append((score, chunk))
                    scored.sort(reverse=True)

                    # Take top 4 chunks as context
                    best = [chunk for _, chunk in scored[:4]]

                    # ---- STEP 5: BUILD CONTEXT ----
                    context = "\n\n---\n\n".join(best)

                    # Include recent chat history for memory
                    history = ""
                    for m in st.session_state.messages[-4:]:
                        role = "User" if m["role"] == "user" else "Assistant"
                        history += f"{role}: {m['content']}\n"

                    # ---- STEP 6: GENERATE ANSWER ----
                    ans = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{
                            "role": "system",
                            "content": (
                                "You are DocMind AI, an expert document assistant. "
                                "Answer questions using ONLY the provided context. "
                                "If the answer is not in the context, say "
                                "'I couldn't find this in the document.' "
                                "Be clear, precise, and helpful."
                            )
                        }, {
                            "role": "user",
                            "content": (
                                f"Conversation History:\n{history}\n\n"
                                f"Document Context:\n{context}\n\n"
                                f"Question: {question}\n\nAnswer:"
                            )
                        }],
                        temperature=0.1,    # Low = more accurate, less creative
                        max_tokens=800
                    )
                    answer = ans.choices[0].message.content

                # Save to chat history
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                st.rerun()

    # ---- ANALYTICS ----
    elif "Analytics" in page:
        st.markdown("""
        <div style='font-family:Syne,sans-serif; font-size:1.8rem;
                    font-weight:800; color:white; margin-bottom:1.5rem;'>
            📊 Analytics
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Faithfulness", "0.9/1.0", "+0.4")
            st.metric("Relevancy", "1.0/1.0", "+0.5")
        with c2:
            st.metric("Recall", "1.0/1.0", "+0.4")
            st.metric("Overall Score", "83%", "+33%")

    # ---- SETTINGS ----
    elif "Settings" in page:
        st.markdown("""
        <div style='font-family:Syne,sans-serif; font-size:1.8rem;
                    font-weight:800; color:white; margin-bottom:1.5rem;'>
            ⚙️ Settings
        </div>""", unsafe_allow_html=True)

        from payments import check_subscription, create_checkout_session
        sub = check_subscription(st.session_state.email)

        # Account info
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);
                    border:1px solid rgba(255,255,255,0.08);
                    border-radius:16px; padding:1.5rem;
                    margin-bottom:1rem;'>
            <div style='color:white; font-weight:600; margin-bottom:1rem;'>
                👤 Account Info
            </div>
            <div style='color:rgba(255,255,255,0.5);
                        font-size:0.9rem; line-height:1.8;'>
                Name: {st.session_state.username}<br>
                Email: {st.session_state.email}<br>
                Status: ✅ Verified
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Subscription status
        if sub["active"]:
            st.markdown("""
            <div style='background:rgba(16,185,129,0.1);
                        border:1px solid rgba(16,185,129,0.3);
                        border-radius:16px; padding:1.5rem;
                        margin-bottom:1rem;'>
                <div style='color:#10b981; font-weight:600; font-size:1.1rem;'>
                    ✅ PRO Plan Active
                </div>
                <div style='color:rgba(255,255,255,0.5);
                            font-size:0.85rem; margin-top:0.5rem;'>
                    $29/month • Unlimited PDFs • All features
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Cancel Subscription"):
                from payments import cancel_subscription
                result = cancel_subscription(st.session_state.email)
                if result["success"]:
                    st.success("Subscription cancelled!")
                else:
                    st.error(result["error"])
        else:
            st.markdown("""
            <div style='background:rgba(99,102,241,0.1);
                        border:1px solid rgba(99,102,241,0.3);
                        border-radius:16px; padding:1.5rem;
                        margin-bottom:1rem;'>
                <div style='color:white; font-weight:600; font-size:1.1rem;'>
                    🆓 Free Plan
                </div>
                <div style='color:rgba(255,255,255,0.5);
                            font-size:0.85rem; margin-top:0.5rem;'>
                    3 PDFs max • 10 questions/day
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style='background:linear-gradient(135deg,
                        rgba(99,102,241,0.15),rgba(6,182,212,0.15));
                        border:1px solid rgba(99,102,241,0.3);
                        border-radius:16px; padding:1.5rem;
                        margin-bottom:1rem;'>
                <div style='color:white; font-weight:700;
                            font-size:1.2rem; margin-bottom:0.5rem;'>
                    ⚡ Upgrade to PRO
                </div>
                <div style='color:rgba(255,255,255,0.6);
                            font-size:0.85rem; margin-bottom:1rem;'>
                    ✅ Unlimited PDFs<br>
                    ✅ Unlimited questions<br>
                    ✅ Priority support<br>
                    ✅ Advanced features
                </div>
                <div style='color:white; font-size:1.5rem; font-weight:800;'>
                    $29<span style='font-size:0.9rem;
                    color:rgba(255,255,255,0.5);'>/month</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("⚡ Upgrade to PRO →", use_container_width=True):
                result = create_checkout_session(
                    st.session_state.email,
                    st.session_state.uid
                )
                if result["success"]:
                    st.markdown(f"[Click here to pay →]({result['url']})")
                    st.info("Click the link above to complete payment!")
                else:
                    st.error("Payment error! Try again!")
