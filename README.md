# Multi-Source Stateful RAG Agent 🤖

A production-ready Retrieval-Augmented Generation (RAG) system that transforms static portfolio documents into an interactive, stateful AI assistant. Implements a **search-before-answer** architecture using LangGraph for decision logic and FAISS for high-speed semantic retrieval.

---

## Key Features

- **Stateful Orchestration** — Uses LangGraph to manage conversation state and enforce retrieval before response generation
- **Torch-Free Architecture** — Optimized for Windows by replacing PyTorch with FastEmbed (ONNX) and Google Generative AI
- **Semantic Memory** — FAISS-based vector search over unstructured `.txt` data with sub-second latency
- **Smart Chunking** — Recursive text splitting (~500 characters with overlap) to preserve semantic context
- **High-Performance Inference** — Integrated with Groq (LLaMA 3.1 / 3.3) for low-latency responses

---

## Technical Stack

| Layer | Technology |
|---|---|
| Frameworks | LangChain, LangGraph |
| LLM | Groq (LLaMA-3.1-8B, LLaMA-3.3-70B) |
| Embeddings | FastEmbed (ONNX, local) / Google Generative AI |
| Vector Database | FAISS |
| Frontend | Streamlit |
| Environment | Python 3.12, uv |

---

## System Architecture

```
User Input (Streamlit)
       ↓
Decision Layer (LangGraph)
       ↓
search_portfolio Tool (FAISS semantic search)
       ↓
LLM Response Generation (Groq)
```

1. **Ingestion** — Loads `.txt` files from the `portfolio/` directory
2. **Indexing** — Applies chunking, generates embeddings, stores in FAISS
3. **Agent Workflow** — LangGraph routes each query through retrieval before generation

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com//Multi-Source-Stateful-RAG-Agent
cd Multi-Source-Stateful-RAG-Agent
```

### 2. Create virtual environment

```bash
uv venv --python 3.12
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
uv pip install streamlit langchain-community langchain-groq fastembed faiss-cpu python-dotenv langgraph
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key   # Optional
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Challenges & Solutions

| Challenge | Solution |
|---|---|
| Windows DLL errors (WinError 1114/126) | Replaced PyTorch with ONNX-based embeddings via FastEmbed |
| Infinite agent loops | Added recursion limits and stricter system prompts |
| Context window overflow | Sliding-window chunking within token limits |

---

## Impact

- Enabled interactive querying over static portfolio data
- Achieved fast, accurate responses via semantic retrieval
- Built a production-style RAG pipeline suitable for real-world deployment

---

## Publication

> Springer / ICDAM 2025 — *Visual Question Answering with Satellite Imagery*
> DOI: [10.1007/978-3-032-04222-4_39](https://doi.org/10.1007/978-3-032-04222-4_39)
