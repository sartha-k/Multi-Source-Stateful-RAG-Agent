Multi-Source Stateful RAG Agent 🤖

A production-ready Retrieval-Augmented Generation (RAG) system that transforms static portfolio documents into an interactive, stateful AI assistant. Implements a “search-before-answer” architecture using LangGraph for decision logic and FAISS for high-speed semantic retrieval.

Key Features
Stateful Orchestration: Uses LangGraph to manage conversation state and enforce retrieval before response generation
Torch-Free Architecture: Optimized for Windows by removing PyTorch dependency and using FastEmbed (ONNX) and Google Generative AI
Semantic Memory: FAISS-based vector search over unstructured .txt data with sub-second latency
Smart Chunking: Recursive text splitting (~500 characters with overlap) to preserve semantic context
High-Performance Inference: Integrated with Groq (LLaMA 3.1 / 3.3) for low-latency responses
Technical Stack

Frameworks: LangChain, LangGraph
LLM: Groq (LLaMA-3.1-8B, LLaMA-3.3-70B)
Embeddings: FastEmbed (ONNX, local) / Google Generative AI
Vector Database: FAISS
Frontend: Streamlit
Environment: Python 3.12, uv package manager

System Architecture
Ingestion: Loads text files from the portfolio/ directory
Indexing: Applies chunking, generates embeddings, and stores them in FAISS
Agent Workflow:
User input via Streamlit
Decision layer checks if retrieval is needed
Calls search_portfolio tool for semantic search
LLM generates response using retrieved context
Installation & Setup
git clone https://github.com/<your-username>/Multi-Source-Stateful-RAG-Agent
cd Multi-Source-Stateful-RAG-Agent
uv venv --python 3.12
.venv\Scripts\activate
uv pip install streamlit langchain-community langchain-groq fastembed faiss-cpu python-dotenv langgraph

Create a .env file:

GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key  # Optional

Run the app:

streamlit run app.py
Challenges & Solutions
Fixed Windows DLL errors (WinError 1114/126) by replacing PyTorch with ONNX-based embeddings
Prevented infinite agent loops using recursion limits and stricter system prompts
Optimized context handling using sliding-window chunking within token limits

Impact
Enabled interactive querying over static portfolio data
Achieved fast and accurate responses using semantic retrieval
Built a production-style RAG pipeline suitable for real-world applications
