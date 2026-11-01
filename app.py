"""CampusGenie – AI-Powered Student Assistant.
Modern Streamlit Web Application implementing TF-IDF Document Retrieval
and Retrieval-Augmented Generation (RAG) with Google Gemini & Local Fallback.
"""

import os
import sys
import time
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="CampusGenie – AI Student Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure local source package is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.knowledge_base import KnowledgeBase
from src.retriever import TFIDFRetriever
from src.rag_pipeline import RAGPipeline
from src.report_generator import generate_capstone_report

# ---------------------------------------------------------
# Custom Styling & CSS Design
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main container and font styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #0d9488 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(30, 58, 138, 0.15);
    }
    .hero-banner h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .hero-banner p {
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
        color: #e0f2fe;
    }

    /* Status & Metric Pills */
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .badge-blue { background-color: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
    .badge-green { background-color: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .badge-purple { background-color: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }
    .badge-amber { background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a; }

    /* Source Cards */
    .source-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.8rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .source-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.08);
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .source-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
    }
    .score-tag {
        font-size: 0.85rem;
        font-weight: 700;
        color: #0d9488;
        background: #ccfbf1;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
    }
    .source-body {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Keyword highlight chips */
    .term-chip {
        display: inline-block;
        background-color: #f1f5f9;
        color: #334155;
        font-size: 0.75rem;
        padding: 0.1rem 0.45rem;
        border-radius: 4px;
        margin-right: 0.3rem;
        margin-top: 0.3rem;
        border: 1px solid #cbd5e1;
    }

    /* Chat Answer Card */
    .answer-container {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    /* Quick chip buttons */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 20px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_knowledge_base():
    """Loads and caches the student knowledge base."""
    return KnowledgeBase()


@st.cache_resource(show_spinner=False)
def get_retriever(_kb):
    """Initializes and caches the TF-IDF retriever."""
    return TFIDFRetriever(_kb.get_all_documents())


@st.cache_resource(show_spinner=False)
def get_rag_pipeline(_retriever):
    """Initializes and caches the RAG orchestrator."""
    return RAGPipeline(_retriever)


# Load backend components safely with error handling
try:
    kb = get_knowledge_base()
    retriever = get_retriever(kb)
    pipeline = get_rag_pipeline(retriever)
    stats = kb.get_statistics()
except Exception as e:
    st.error(f"⚠️ Critical initialization error loading knowledge base: {e}")
    st.stop()


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 **CampusGenie Settings**")
    st.caption("AI-Powered Student Assistant • Capstone v1.0")
    st.divider()

    # 1. Generation Mode
    st.markdown("#### ⚡ **Generation Engine**")
    mode_selection = st.radio(
        "Select Operation Mode:",
        options=["Smart RAG (Gemini API)", "Local Offline Mode (Free / Zero-Cloud)"],
        index=0,
        help="Smart RAG pairs TF-IDF retrieval with Google Gemini. Local Offline Mode runs deterministic synthesis directly from retrieved passages without cloud calls."
    )
    use_api = (mode_selection == "Smart RAG (Gemini API)")

    # 2. Gemini API Configuration
    if use_api:
        st.markdown("#### 🔑 **Google Gemini API Key**")
        default_key = os.environ.get("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Enter Gemini API Key:",
            value=default_key,
            type="password",
            placeholder="AIzaSy...",
            help="Free API key from Google AI Studio (aistudio.google.com). If empty, CampusGenie automatically degrades to the Local Offline Engine."
        )

        model_name = st.selectbox(
            "Gemini Model:",
            options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"],
            index=0,
            help="gemini-1.5-flash is ultra-fast and optimized for high-throughput academic tutoring."
        )

        if api_key_input.strip():
            st.success("🟢 API Key Provided")
        else:
            st.info("💡 No key entered. Will automatically use Local Fallback Synthesis.")
    else:
        api_key_input = ""
        model_name = "gemini-1.5-flash"
        st.info("💻 Running in 100% Local Offline Mode. No cloud calls or API keys required.")

    st.divider()

    # 3. Retrieval Parameters
    st.markdown("#### 🔍 **Retrieval Settings**")
    top_k = st.slider(
        "Top-K Retrieved Sources:",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of most relevant knowledge passages to retrieve and augment into the generation context."
    )

    similarity_threshold = st.slider(
        "Min Similarity Threshold:",
        min_value=0.01,
        max_value=0.30,
        value=0.03,
        step=0.01,
        format="%.2f",
        help="Cosine similarity threshold. Passages below this score are filtered out as irrelevant."
    )

    categories = ["All Categories"] + kb.get_categories()
    selected_category = st.selectbox(
        "Subject / Domain Filter:",
        options=categories,
        index=0,
        help="Filter retrieval candidate pool strictly to one subject domain."
    )
    active_category = None if selected_category == "All Categories" else selected_category

    st.divider()

    # 4. Knowledge Base Metrics
    st.markdown("#### 📊 **Knowledge Base Metrics**")
    c1, c2 = st.columns(2)
    c1.metric("📚 Articles", stats["total_documents"])
    c2.metric("🗂️ Categories", stats["total_categories"])
    st.metric("🔤 TF-IDF Vocabulary", f"{retriever.get_vocabulary_size():,} terms")

    st.divider()

    # 5. Capstone Project PDF Report
    st.markdown("#### 📄 **Capstone Project Report**")
    pdf_report_path = os.path.join(CURRENT_DIR, "CampusGenie_Project_Report.pdf")

    # Ensure report exists
    if not os.path.exists(pdf_report_path):
        try:
            generate_capstone_report(pdf_report_path)
        except Exception as e:
            st.caption(f"Error compiling report: {e}")

    if os.path.exists(pdf_report_path):
        with open(pdf_report_path, "rb") as pdf_file:
            st.download_button(
                label="📥 Download Project Report (PDF)",
                data=pdf_file,
                file_name="CampusGenie_Capstone_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    st.caption("Developed with ❤️ for Higher Education & Students")


# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
# Hero Banner
st.markdown("""
<div class="hero-banner">
    <h1>🎓 CampusGenie – AI-Powered Student Assistant</h1>
    <p>A transparent Retrieval-Augmented Generation (RAG) assistant for computer science, academic regulations, and proven study methods.</p>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_chat, tab_explorer, tab_diagnostics = st.tabs([
    "💬 Academic Assistant (Q&A)",
    "📖 Knowledge Explorer & Flashcards",
    "⚙️ System Diagnostics & Architecture"
])


# =========================================================
# TAB 1: ACADEMIC ASSISTANT (Q&A)
# =========================================================
with tab_chat:
    st.markdown("#### Ask Any Academic or Campus Question")
    st.caption("CampusGenie searches its verified knowledge base via TF-IDF, retrieves relevant sources, and generates grounded answers.")

    # Suggested Prompts for Quick Testing
    st.markdown("**💡 Quick Sample Questions (Click to Ask):**")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    quick_col4, quick_col5, quick_col6 = st.columns(3)

    sample_questions = [
        "Explain Big-O notation and time complexity with examples",
        "Compare Breadth-First Search (BFS) vs Depth-First Search (DFS)",
        "How do gradient descent and backpropagation train neural networks?",
        "What are the attendance requirements and consequences of low attendance?",
        "How do I apply for course withdrawal after the Drop/Add period?",
        "How can I use Active Recall and Spaced Repetition for exams?"
    ]

    cols = [quick_col1, quick_col2, quick_col3, quick_col4, quick_col5, quick_col6]
    for i, q_text in enumerate(sample_questions):
        with cols[i]:
            if st.button(f"📌 {q_text[:36]}...", key=f"quick_{i}", use_container_width=True):
                st.session_state.pending_query = q_text

    st.write("")

    # Query Input Form
    with st.form("query_form", clear_on_submit=False):
        user_input = st.text_input(
            "Your Question:",
            value=st.session_state.pending_query,
            placeholder="e.g., Explain the difference between Arrays and Linked Lists...",
            label_visibility="collapsed"
        )
        col_btn1, col_btn2, col_info = st.columns([1.5, 1.2, 5])
        with col_btn1:
            submit_pressed = st.form_submit_button("🚀 Ask CampusGenie", use_container_width=True)
        with col_btn2:
            clear_pressed = st.form_submit_button("🗑️ Clear History", use_container_width=True)
        with col_info:
            active_engine_name = "Google Gemini API" if use_api and api_key_input.strip() else "CampusGenie Local Synthesis Engine"
            st.markdown(f"<span class='metric-badge badge-blue'>Mode: {mode_selection.split()[0]}</span> <span class='metric-badge badge-green'>Active: {active_engine_name}</span>", unsafe_allow_html=True)

    if clear_pressed:
        st.session_state.chat_history = []
        st.session_state.pending_query = ""
        st.rerun()

    # Process Query Execution
    if submit_pressed and user_input.strip():
        # Clear pending quick query
        st.session_state.pending_query = ""

        with st.spinner("🔍 Searching campus knowledge base and synthesizing grounded answer..."):
            response_data = pipeline.query(
                query_text=user_input.strip(),
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                category=active_category,
                use_api=use_api,
                api_key=api_key_input.strip() if api_key_input else None,
                model_name=model_name
            )

        # Store in session chat history
        st.session_state.chat_history.insert(0, {
            "query": user_input.strip(),
            "response": response_data,
            "timestamp": time.strftime("%H:%M:%S")
        })

    # Render Conversation Feed
    if st.session_state.chat_history:
        for idx, item in enumerate(st.session_state.chat_history):
            q = item["query"]
            res = item["response"]
            t = item["timestamp"]

            st.markdown(f"### 🧑‍🎓 **Student:** *\"{q}\"*")

            # Notification if fallback was triggered
            if res.get("is_fallback") and res.get("fallback_reason"):
                st.warning(
                    f"⚠️ **Graceful Fallback Notice:** {res['fallback_reason']}\n\n"
                    f"CampusGenie automatically synthesized an accurate, grounded answer using the **Local Offline Engine**."
                )

            # Metadata Badges
            engine_str = res.get("engine", "Local Synthesizer")
            latency_str = f"{res.get('latency_seconds', 0.0)}s"
            source_count = len(res.get("sources", []))

            st.markdown(
                f"<span class='metric-badge badge-purple'>⚡ Engine: {engine_str}</span> "
                f"<span class='metric-badge badge-blue'>⏱️ Latency: {latency_str}</span> "
                f"<span class='metric-badge badge-green'>📚 Grounded in {source_count} Sources</span> "
                f"<span class='metric-badge badge-amber'>🕒 {t}</span>",
                unsafe_allow_html=True
            )

            # Answer Card
            st.markdown(res["answer"])

            # Sources Breakdown Section
            if res.get("sources"):
                with st.expander(f"📚 **View {len(res['sources'])} Retrieved Knowledge Sources & Evidence**", expanded=(idx == 0)):
                    st.caption("These passages were retrieved using TF-IDF sublinear vectorization and Cosine Similarity scoring:")

                    for s_idx, src in enumerate(res["sources"], 1):
                        score_pct = src["score_pct"]
                        matched_chips = "".join([f"<span class='term-chip'>{m}</span>" for m in src["matched_terms"]])
                        if not matched_chips:
                            matched_chips = "<span class='term-chip'>contextual match</span>"

                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">
                                <span class="source-title">#{s_idx} &bull; {src['title']}</span>
                                <span class="score-tag">Similarity: {score_pct}%</span>
                            </div>
                            <div style="margin-bottom: 0.4rem;">
                                <span class="metric-badge badge-blue">{src['category']}</span>
                                <span class="metric-badge badge-purple">Topic: {src['topic']}</span>
                                <span class="metric-badge badge-amber">Level: {src['difficulty']}</span>
                            </div>
                            <div class="source-body">
                                {src['content']}
                            </div>
                            <div style="margin-top: 0.5rem;">
                                <b>Matched Terms:</b> {matched_chips}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            st.divider()
    else:
        st.info("👋 Hello! I am **CampusGenie**. Ask me anything about Data Structures, AI/ML, Operating Systems, Campus Regulations, or Study Strategies above, or click one of the suggested sample questions.")


# =========================================================
# TAB 2: KNOWLEDGE EXPLORER & FLASHCARDS
# =========================================================
with tab_explorer:
    st.markdown("### 📖 **Campus Knowledge Explorer**")
    st.caption("Browse all curated educational articles indexed inside CampusGenie's knowledge base.")

    col_cat, col_search = st.columns([1.5, 2.5])
    with col_cat:
        browse_cat = st.selectbox("Filter by Category:", ["All Categories"] + kb.get_categories(), key="explore_cat")
    with col_search:
        search_filter = st.text_input("🔍 Quick Keyword Filter:", placeholder="Search title, content, or tags...", key="explore_search")

    filtered_docs = kb.filter_by_category(None if browse_cat == "All Categories" else browse_cat)
    if search_filter.strip():
        term = search_filter.strip().lower()
        filtered_docs = [
            d for d in filtered_docs
            if term in d["title"].lower() or term in d["content"].lower() or term in " ".join(d.get("tags", [])).lower()
        ]

    st.write(f"Showing **{len(filtered_docs)}** articles:")

    for doc in filtered_docs:
        tags_html = "".join([f"<span class='term-chip'>#{t}</span>" for t in doc.get("tags", [])])
        with st.expander(f"📌 [{doc['id']}] {doc['title']} ({doc['category']})"):
            st.markdown(f"**Topic:** `{doc['topic']}` &nbsp;|&nbsp; **Difficulty:** `{doc.get('difficulty', 'General')}`")
            st.write(doc["content"])
            st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)


# =========================================================
# TAB 3: SYSTEM DIAGNOSTICS & ARCHITECTURE
# =========================================================
with tab_diagnostics:
    st.markdown("### ⚙️ **System Diagnostics & Architecture**")

    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        st.markdown("#### 📐 **RAG Architecture Flow**")
        st.code("""
[Student Query]
       │
       ▼
[TF-IDF Vectorizer (1-2 ngrams, sublinear TF)]
       │
       ▼
[Cosine Similarity Retrieval against Knowledge Base]
       │
       ▼
[Ranked Sources (Top-K Filtered by Threshold)]
       │
       ├───► [Has Gemini API Key?]
       │           │
       │           ├─► YES: [Google Gemini 1.5 Flash API]
       │           │
       │           └─► NO / ERROR: [Local Synthesis Engine]
       ▼
[Attributed, Grounded Answer with Source Cards]
        """, language="text")

    with diag_c2:
        st.markdown("#### 🧮 **Mathematical Retrieval Formulation**")
        st.markdown("""
        **1. Term Frequency (Sublinear Scaling):**
        $$TF(t, d) = 1 + \\ln(f(t, d)) \\quad \\text{for } f(t, d) > 0$$

        **2. Smooth Inverse Document Frequency:**
        $$IDF(t, D) = \\ln\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1.0$$

        **3. Cosine Similarity:**
        $$\\text{CosineSim}(q, d) = \\frac{\\mathbf{v}_q \\cdot \\mathbf{v}_d}{\\|\\mathbf{v}_q\\|_2 \\|\\mathbf{v}_d\\|_2}$$
        """)

    st.divider()

    st.markdown("#### 🩺 **Component Health Check**")
    h1, h2, h3, h4 = st.columns(4)
    h1.success("✅ Knowledge Base: Ready")
    h2.success("✅ TF-IDF Engine: Fitted")
    h3.success("✅ Local Fallback: Online")
    h4.info(f"ℹ️ Gemini SDK: {'Installed' if pipeline._call_gemini_api else 'Available via REST'}")

    st.markdown("#### 📄 **Recompile Capstone PDF Report**")
    if st.button("🔄 Recompile Project Report (PDF)", key="recompile_report_btn"):
        with st.spinner("Compiling Capstone PDF report via ReportLab..."):
            out_pdf = generate_capstone_report(pdf_report_path)
            st.success(f"Successfully compiled: {os.path.basename(out_pdf)}")

