# ============================================
# DOCMIND AI — International Level SaaS
# Firebase Auth + Hybrid RAG + Beautiful UI
# ============================================
import streamlit as st
from auth import signup_user, login_user, verify_token
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import chromadb
from groq import Groq
import tempfile
import os

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

.auth-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 25px 50px rgba(0,0,0,0.5);
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
# SESSION STATE
# ============================================
for key, val in {
    "logged_in": False,
    "token": None,
    "username": "",
    "email": "",
    "uid": "",
    "messages": [],
    "pdf_ready": False,
    "collection": None,
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

        # Tab buttons
        t1, t2 = st.columns(2)
        with t1:
            if st.button("Sign In",
                use_container_width=True,
                type="primary" if st.session_state.auth_mode == "login"
                else "secondary"):
                st.session_state.auth_mode = "login"
                st.rerun()
        with t2:
            if st.button("Sign Up",
                use_container_width=True,
                type="primary" if st.session_state.auth_mode == "signup"
                else "secondary"):
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

            email = st.text_input(
                "Email", placeholder="Enter your email"
            )
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

            name = st.text_input(
                "Full Name", placeholder="Your name"
            )
            email = st.text_input(
                "Email", placeholder="your@email.com"
            )
            password = st.text_input(
                "Password", type="password",
                placeholder="Min 6 characters"
            )
            confirm = st.text_input(
                "Confirm Password", type="password",
                placeholder="Repeat password"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(
                "Create Account →",
                use_container_width=True
            ):
                if not all([name, email, password, confirm]):
                    st.warning("Please fill all fields!")
                elif len(password) < 6:
                    st.error("Password must be 6+ characters!")
                elif password != confirm:
                    st.error("Passwords do not match!")
                else:
                    with st.spinner("Creating account..."):
                        result = signup_user(
                            email, password, name
                        )
                    if result["success"]:
                        st.success(
                            "Account created! Please sign in!"
                        )
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
                    <div style='color:white; font-weight:600;
                                font-size:0.9rem;'>
                        {st.session_state.username}
                    </div>
                    <div style='color:rgba(255,255,255,0.4);
                                font-size:0.75rem;'>
                        {st.session_state.email}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "",
            ["🏠  Dashboard",
             "💬  Chat",
             "📊  Analytics",
             "⚙️  Settings"],
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
                                text-transform:uppercase;
                                letter-spacing:1px;'>
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

        if not st.session_state.pdf_ready:
            uploaded = st.file_uploader(
                "Upload PDF", type="pdf",
                label_visibility="collapsed"
            )
            if uploaded:
                with st.spinner("⚡ Processing..."):
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500, chunk_overlap=50
                    )
                    chunks = splitter.split_documents(pages)
                    texts = [
                        c.page_content for c in chunks
                        if c.page_content.strip()
                    ]

                    db = chromadb.EphemeralClient()
try:
    db.delete_collection("pdf_docs")
except:
    pass
col = db.create_collection("pdf_docs")
col.add(
    documents=texts,
    ids=[f"c_{i}" for i in range(len(texts))]
)
st.session_state.collection = col
st.session_state.all_texts = texts
                    st.session_state.pdf_ready = True
                    st.session_state.pdf_name = uploaded.name
                st.rerun()
        else:
            st.success(f"✅ {st.session_state.pdf_name} ready!")

            if st.button("📄 Upload New PDF"):
                st.session_state.pdf_ready = False
                st.session_state.messages = []
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # Show messages
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

            question = st.chat_input(
                "Ask anything about your document..."
            )

            if question:
                groq_client = Groq(
                    api_key=os.environ.get("GROQ_API_KEY")
                )
                with st.spinner("⚡ Thinking..."):
    # Multi query
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content":
            f"Generate 3 search queries for: {question}\n"
            f"Return 3 questions only, one per line."}]
    )
    queries = r.choices[0].message.content.strip().split("\n")
    queries.append(question)

    # Hybrid search
    all_chunks = []
    for q in queries:
        # Vector search via ChromaDB
        vr = st.session_state.collection.query(
            query_texts=[q], n_results=3
        )
        all_chunks.extend(vr["documents"][0])

        # BM25 keyword search
        tokenized = [
            t.lower().split()
            for t in st.session_state.all_texts
        ]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(q.lower().split())
        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:3]
        all_chunks.extend([
            st.session_state.all_texts[i]
            for i in top_idx
        ])

    # Remove duplicates
    unique = list(set(all_chunks))

    # Simple keyword reranking - no torch needed!
    q_words = set(question.lower().split())
    scored = []
    for chunk in unique:
        c_words = set(chunk.lower().split())
        score = len(q_words & c_words) / max(
            len(q_words | c_words), 1
        )
        scored.append((score, chunk))
    scored.sort(reverse=True)
    best = [c for _, c in scored[:3]]

                    context = "\n\n".join(best)
                    history = ""
                    for m in st.session_state.messages[-4:]:
                        history += f"{m['role'].upper()}: {m['content']}\n"

                    ans = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content":
                            f"You are DocMind AI assistant.\n"
                            f"Answer using ONLY context below.\n"
                            f"History: {history}\n"
                            f"Context: {context}\n"
                            f"Question: {question}\nAnswer:"}]
                    )
                    answer = ans.choices[0].message.content

                st.session_state.messages.append({
                    "role": "user", "content": question
                })
                st.session_state.messages.append({
                    "role": "assistant", "content": answer
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

        # ============================================
        # CHECK SUBSCRIPTION STATUS
        # ============================================
        from payments import check_subscription, create_checkout_session
        sub = check_subscription(st.session_state.email)

        # Show account info
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);
                    border:1px solid rgba(255,255,255,0.08);
                    border-radius:16px; padding:1.5rem;
                    margin-bottom:1rem;'>
            <div style='color:white; font-weight:600;
                        margin-bottom:1rem;'>
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

        # Show subscription status
        if sub["active"]:
            st.markdown("""
            <div style='background:rgba(16,185,129,0.1);
                        border:1px solid rgba(16,185,129,0.3);
                        border-radius:16px; padding:1.5rem;
                        margin-bottom:1rem;'>
                <div style='color:#10b981; font-weight:600;
                            font-size:1.1rem;'>
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
                result = cancel_subscription(
                    st.session_state.email
                )
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
                <div style='color:white; font-weight:600;
                            font-size:1.1rem;'>
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
                <div style='color:white; font-size:1.5rem;
                            font-weight:800;'>
                    $29<span style='font-size:0.9rem;
                    color:rgba(255,255,255,0.5);'>/month</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(
                "⚡ Upgrade to PRO →",
                use_container_width=True
            ):
                result = create_checkout_session(
                    st.session_state.email,
                    st.session_state.uid
                )
                if result["success"]:
                    st.markdown(
                        f"[Click here to pay →]({result['url']})"
                    )
                    st.info(
                        "Click the link above to complete payment!"
                    )
                else:
                    st.error("Payment error! Try again!")
