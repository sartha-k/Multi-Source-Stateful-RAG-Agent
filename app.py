

# %%
from dotenv import load_dotenv
import os

# This looks for the .env file in the exact same folder as this script
load_dotenv(os.path.join(os.getcwd(), '.env')) 


# %%
import os
import glob
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCS_DIR = "portfolio"
# Create the directory if it doesn't exist so glob doesn't fail
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# 1. The Function Definition
def load_portfolio(main_dir):
    docs = [] # Renamed to 'docs' for clarity
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                category = os.path.basename(root)
                docs.append(Document(
                    page_content=text, 
                    metadata={"source": file, "category": category}
                ))
    return docs

# 2. The Execution (The correct order)
DOCS_DIR = "portfolio"
raw_docs = load_portfolio(DOCS_DIR) 
print(f"Loaded {len(raw_docs)} raw documents.")

# 3. Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, 
    chunk_overlap=50
)
chunks = text_splitter.split_documents(raw_docs) # Final 'chunks' variable

# 4. Final check
print(f"DEBUG: Successfully created {len(chunks)} small chunks for searching.")




# %%
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# This runs 100% on your laptop. No API keys, no rate limits, and it's very fast.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")






# %%
import os
#import google.generativeai as genai
from dotenv import load_dotenv




# %%
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# 1. Use the EXACT model name that was FOUND
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_document"
)
# creating chunks
if chunks:
    if os.path.exists("faiss_index"):
        print("Loading existing index...")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new index...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local("faiss_index")

    
    



# %%
import os
import time
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 1. LOAD API KEYS
load_dotenv()

# 2. DEFINE THE SEARCH TOOL
@tool
def search_portfolio(query: str) -> str:
    """Searches the portfolio for resume and project details."""
    # 1. Access the vector_store already created/loaded in the main app
    # DO NOT re-load the index inside the function; it makes it very slow.
    global vector_store
    
    try:
        if vector_store is None:
            return "Error: Vector store not initialized."
            
        # 2. Perform the search
        docs = vector_store.similarity_search(query, k=2)
        
        if not docs:
            return "No relevant information found in the documents."
            
        # 3. Format and return the results
        return "\n\n".join([doc.page_content for doc in docs])
        
    except Exception as e:
        return f"Error during search: {e}"


tools = [search_portfolio]
tool_node = ToolNode(tools)

# 3. CONFIGURE LLM (GROQ)
# Binding tools tells the LLM it has access to the search function
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 4. DEFINE STATE SCHEMA
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 5. SYSTEM PROMPT (Crucial for fixing Error 400)
SYSTEM_PROMPT = SystemMessage(content=(
    "You are a professional Career Assistant representing [Your Name]. "
    "Your goal is to answer questions based ONLY on the provided Resume, Projects, and Research documents. "
    "When asked about skills, experience, or specific projects, you MUST use the search_portfolio tool. "
    "Be professional, concise, and highlight technical achievements. "
    "If the information is not found in the documents, politely state that you don't have that information."
))

# 6. DEFINE GRAPH NODES
def call_model(state: AgentState):
    messages = state["messages"]
    # Ensure system instructions are always present
    if not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def route_tools(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

# 7. BUILD THE WORKFLOW
builder = StateGraph(AgentState)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_tools)
builder.add_edge("tools", "agent")

# 8. COMPILE WITH MEMORY (For conversation history)
memory = MemorySaver()
agent_app = builder.compile(checkpointer=memory)

# %%
from langgraph.checkpoint.memory import MemorySaver

# 1. Initialize Memory
memory = MemorySaver()

# 2. Re-compile the graph with memory
# (Using your 'builder' from the previous step)
app = builder.compile(checkpointer=memory)

# %%
from langgraph.graph import StateGraph, START, END

# 1. Start fresh (this clears the previous 'already exists' errors)
builder = StateGraph(AgentState)

# 2. Add your Nodes
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

# 3. Define the Flow (The "Roads")
builder.add_edge(START, "agent")

# Add the conditional path
builder.add_conditional_edges(
    "agent",
    route_tools,
    {
        "tools": "tools", 
        END: END
    }
)

# Crucial: Connect tools back to agent so it can use the search results
builder.add_edge("tools", "agent")

# 4. Compile
memory = MemorySaver()
agent_app = builder.compile(checkpointer=memory)

# ... (all previous code: imports, tool, agent_app setup) ...

# 4. STREAMLIT UI SESSION STATE
import streamlit as st
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.write(msg.content)

# 10. THE EXECUTION BLOCK (PUT THIS AT THE VERY END)
# --- 1. SESSION STATE (Memory for the Web Page) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. CHAT INPUT ---
if prompt := st.chat_input("Ask me about my experience or projects..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 4. RUN AGENT ---
    with st.chat_message("assistant"):
        with st.spinner("Searching docs..."):
            # Prepare the input for the agent
            config = {"configurable": {"thread_id": "st_user_1"}}
            
            # Use .invoke to get the final answer
            result = agent_app.invoke({"messages": [("user", prompt)]}, config)
            full_response = result["messages"][-1].content
            
            st.markdown(full_response)
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})




