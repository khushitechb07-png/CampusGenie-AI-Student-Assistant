# 🎓 CampusGenie – AI-Powered Student Assistant

> **A Complete Beginner-Friendly Generative AI Capstone Project**  
> *Combining Explainable TF-IDF Document Retrieval, Retrieval-Augmented Generation (RAG), Google Gemini API, and Multi-Tier Offline Fallback.*

---

## 🌟 Executive Overview

**CampusGenie** is an end-to-end, full-stack educational AI assistant designed specifically for university students, educators, and AI researchers. It empowers students to ask complex academic queries spanning **Data Structures & Algorithms (DSA)**, **Artificial Intelligence & Machine Learning (AI/ML)**, **Computer Systems & Databases**, **Campus Life & Academic Policies**, and **Proven Study Strategies**.

Unlike generic chatbot interfaces that hallucinate answers or rely blindly on external proprietary clouds, CampusGenie introduces an **explainable Retrieval-Augmented Generation (RAG)** architecture:
1. **Explainable Information Retrieval:** Employs **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** to index and retrieve verified syllabus articles.
2. **Dual-Tier Generation Engine:**
   - **Smart RAG (Online):** Integrates Google's state-of-the-art **Gemini API** (`gemini-1.5-flash`, `gemini-1.5-pro`) to provide fluent, pedagogically structured explanations with inline citations.
   - **Local Offline Engine (Free & Resilient):** When no API key is entered or during internet outages, CampusGenie runs an internal **deterministic extractive synthesis engine** that answers questions directly from the highest-ranked passages with 0% downtime.
3. **Verifiable Source Grounding:** Every response includes collapsible evidence cards displaying exact source titles, subject areas, relevance percentages, and matched keyword tokens.
4. **Academic Capstone PDF Report:** Generates a publication-quality capstone project report formatted according to university standards.

---

## 🏛️ System Architecture

```text
                               ┌────────────────────────────────┐
                               │     Student Academic Query     │
                               └──────────────┬─────────────────┘
                                              │
                                              ▼
                               ┌────────────────────────────────┐
                               │   Query Preprocessor & Parser  │
                               │  (Lowercasing, Stopwords, etc.)│
                               └──────────────┬─────────────────┘
                                              │
                                              ▼
  ┌─────────────────────────┐  TF-IDF Vector   ┌────────────────────────────────┐
  │  Campus Knowledge Base  │ ───────────────► │  Cosine Similarity Matcher     │
  │  (data/campus_knowledge)│                  │  (sublinear TF, bi-grams 1-2)  │
  └─────────────────────────┘                  └──────────────┬─────────────────┘
                                                              │
                                                              ▼
                                               ┌───────────────────────────────┐
                                               │   Top-K Ranked Sources        │
                                               │ (Filtered by Threshold & Cat) │
                                               └──────────────┬────────────────┘
                                                              │
                                       ┌──────────────────────┴──────────────────────┐
                                       │                                             │
                      [Has Gemini API Key?]                                [Offline / Free Mode]
                                       │                                             │
                                       ▼ YES                                         ▼ NO / FALLBACK
                       ┌───────────────────────────────┐             ┌───────────────────────────────┐
                       │  Augmented Academic Prompt    │             │ CampusGenie Local Synthesizer │
                       │  + Google Gemini 1.5 Flash    │             │ (Deterministic Extraction &   │
                       └──────────────┬────────────────┘             │  bulleted takeaways)          │
                                      │                              └──────────────┬────────────────┘
                                      │                                             │
                                      └──────────────────────┬──────────────────────┘
                                                             │
                                                             ▼
                                              ┌───────────────────────────────┐
                                              │  Modern Streamlit Application │
                                              │ - Structured Answer           │
                                              │ - Evidence Cards & Badges     │
                                              │ - Knowledge Base Explorer     │
                                              │ - PDF Report Download         │
                                              └───────────────────────────────┘
```

---

## 🧮 Mathematical Formulation

### 1. Sublinear Term Frequency (TF)
To prevent repeated words within a single document from dominating the search score, CampusGenie applies sublinear TF scaling:
$$TF(t, d) = \begin{cases} 1 + \ln(f(t, d)) & \text{if } f(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 2. Smooth Inverse Document Frequency (IDF)
Penalizes ubiquitous words across the entire campus corpus while assigning higher weight to discriminative technical terms (e.g., *'backpropagation'*, *'asymptotic'*, *'withdrawal'*):
$$IDF(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1.0$$

### 3. Cosine Similarity Ranking
Measures the directional alignment between the normalized query vector $\mathbf{v}_q$ and each document vector $\mathbf{v}_d$:
$$\text{CosineSim}(q, d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\|_2 \, \|\mathbf{v}_d\|_2} = \frac{\sum_{i=1}^{V} w_{q, i} \cdot w_{d, i}}{\sqrt{\sum_{i=1}^{V} w_{q, i}^2} \cdot \sqrt{\sum_{i=1}^{V} w_{d, i}^2}}$$

---

## 📂 Project Directory Structure

```text
CampusGenie/
│
├── app.py                          # Modern Streamlit Web Application
├── run.py                          # Quick Python launcher (python run.py)
├── run.bat                         # One-click Windows batch runner
├── requirements.txt                # Production dependencies
├── README.md                       # Complete documentation & capstone overview
├── CampusGenie_Project_Report.pdf  # Professional Capstone Project Report (PDF)
│
├── src/                            # Modular Backend Package
│   ├── __init__.py                 # Package initializer
│   ├── knowledge_base.py           # Loads, validates & chunks campus knowledge
│   ├── retriever.py                # TF-IDF Vectorizer & Cosine Similarity search engine
│   ├── rag_pipeline.py             # RAG Orchestrator (Gemini API + Local Offline Synthesizer)
│   └── report_generator.py         # Automated ReportLab PDF generator
│
├── data/                           # Curated Student Educational Corpus
│   └── campus_knowledge.json       # JSON dataset covering DSA, AI/ML, OS/DB, Policies, Study Tips
│
└── tests/                          # 100% Automated Test Suite
    ├── __init__.py
    ├── test_knowledge_base.py      # Schema, loading & category filtering tests
    ├── test_retriever.py           # TF-IDF, Cosine similarity, ranking & edge case tests
    ├── test_rag_pipeline.py        # Prompt construction & offline fallback tests
    └── run_all_tests.py            # Master test runner with pass/fail summary
```

---

## 🚀 Quick Setup & Installation Guide

### Prerequisites
- **Python 3.10, 3.11, or 3.12** installed on your system.

### Step 1: Navigate to the CampusGenie Directory
Open your terminal or command prompt:
```bash
cd CampusGenie
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Automated Verification Suite
Verify that all algorithms, schemas, and pipelines are functioning with 100% test coverage:
```bash
python tests/run_all_tests.py
```
*Expected Output:*
```text
======================================================================
CAMPUSGENIE CAPSTONE PROJECT - AUTOMATED TEST SUITE
----------------------------------------------------------------------
Ran 20 tests in 0.187s

OK
[SUMMARY] 20 Tests Run
   [PASS]   Passed:   20
   [FAIL]   Failed:   0
   [RATE]   Success:  100.0%
[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!
======================================================================
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*Or simply execute:*
```bash
python run.py
```
The application will launch automatically in your default browser at:
`http://localhost:8501`

---

## 🔑 Google Gemini API Configuration (Optional)

CampusGenie is designed to work **100% out-of-the-box in Local Offline Mode** without requiring any API keys.

If you wish to enable conversational AI generation via Google Gemini:
1. Obtain a **free API Key** from [Google AI Studio](https://aistudio.google.com/).
2. Enter your key directly in the sidebar input field within the web app, or set it as an environment variable:
   ```bash
   # Windows (PowerShell):
   $env:GEMINI_API_KEY="your_api_key_here"

   # Windows (CMD):
   set GEMINI_API_KEY=your_api_key_here

   # Linux/macOS:
   export GEMINI_API_KEY="your_api_key_here"
   ```
3. If your API key ever encounters rate limits or network issues, CampusGenie will **automatically switch to the Local Synthesis Engine**, ensuring continuous availability.

---

## 📚 Knowledge Base Coverage

| Domain Category | Key Topics & Concepts Covered |
| :--- | :--- |
| **Data Structures & Algorithms** | Big-O Notation, Asymptotic Complexity, Arrays vs. Linked Lists, Binary Search Trees, Self-Balancing Trees (AVL/Red-Black), BFS vs. DFS, Dynamic Programming (Memoization vs Tabulation). |
| **Artificial Intelligence & ML** | Supervised vs. Unsupervised vs. Reinforcement Learning, Bias-Variance Tradeoff, L1/L2 Regularization, Backpropagation & Gradient Descent, Transformers & RAG Architecture. |
| **Computer Systems & Databases** | Processes vs. Threads, Concurrency & Mutexes, ACID Guarantees, Relational vs. NoSQL, B+ Tree Database Indexing and I/O Optimization. |
| **Campus Life & Policies** | Course Registration, Drop/Add Windows, Course Withdrawal ('W' Grade), 75% Mandatory Attendance Rule, GPA Calculation, Academic Integrity & Honor Code. |
| **Study Strategies & Wellness** | Active Recall, Spaced Repetition (Ebbinghaus Forgetting Curve), Pomodoro Technique, Cal Newport's Deep Work, Exam Anxiety & Sleep Hygiene. |

---

## 🧪 Automated Testing & Reliability Engineering

The system includes 20 unit tests across three dedicated test suites:
- `test_knowledge_base.py`: Validates schema integrity, non-empty fields, category extraction, and ID lookups.
- `test_retriever.py`: Evaluates TF-IDF vocabulary construction, ranking monotonicity, category filtering, out-of-vocabulary handling, and whitespace queries.
- `test_rag_pipeline.py`: Verifies context formatting, system prompt engineering, source citations, and deterministic local synthesis fallback.

To execute tests at any time:
```bash
python tests/run_all_tests.py
```

---

## 📄 Capstone Project Report (PDF)

A formal, publication-ready academic report has been pre-compiled and included in the project root:
- **File:** `CampusGenie_Project_Report.pdf`
- **Features:** Executive Summary, Problem Statement, System Architecture, Mathematical Formulas, Test Results, and Future Roadmap.
- **Download:** Directly accessible from the sidebar of the running Streamlit web application.

To recompile the report at any time:
```bash
python src/report_generator.py
```

---

## 🛡️ Error Handling & Fault Tolerance

- **Corpus Validation:** Checks for required JSON keys at startup and raises actionable configuration errors.
- **Graceful API Degradation:** Automatically catches Gemini API quota errors, network timeouts, and missing credentials, switching to local offline synthesis without disrupting the student session.
- **Safe Query Processing:** Protects against empty queries, queries with only stop words, and out-of-vocabulary terms with helpful guidance chips.
- **Platform Agnostic Output:** Ensures Unicode and terminal encoding safety across Windows PowerShell, Linux, and macOS.

---

## 👥 Authors & Academic Credits

- **Project:** CampusGenie – AI-Powered Student Assistant
- **Framework:** Python 3.12, Streamlit, Scikit-Learn, Google Gemini, ReportLab
- **License:** MIT License — Open for academic study, extension, and capstone evaluation.

