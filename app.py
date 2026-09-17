"""
RAG Chatbot - Streamlit Application
=====================================

This is the main entry point for the RAG Chatbot. It provides a web UI where users can:
1. Upload PDF documents
2. Process and embed them into a vector store
3. Chat with the documents using natural language

Run this app with: streamlit run app.py
"""

import streamlit as st
from document_processor import process_pdfs, chunk_documents
from vector_store import create_vector_store, get_vector_store
from rag_engine import generate_answer, get_conversational_answer
from config import CHUNK_SIZE, CHUNK_OVERLAP


# --- Page Configuration ---
st.set_page_config(
    page_title="Archie - Intelligent Document Reader",
    page_icon="📚",
    layout="wide"
)

# --- Session State Initialization ---
# Streamlit reruns the entire script on every interaction.
# Session state persists data across these reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # The loaded vector store
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False  # Whether PDFs have been processed


# --- Sidebar: Document Upload ---
with st.sidebar:
    st.title("📁 Document Upload")
    st.markdown("---")
    
    # File uploader — accepts multiple PDFs
    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat with."
    )
    
    # Process button
    if uploaded_files and st.button("🔄 Process Documents", type="primary"):
        with st.spinner("Processing your documents..."):
            try:
                # Step 1: Extract text from PDFs
                st.info("📖 Extracting text from PDFs...")
                documents = process_pdfs(uploaded_files)
                
                if not documents:
                    st.error("❌ No text could be extracted from the uploaded PDFs.")
                    st.stop()
                
                # Step 2: Chunk the documents
                st.info("✂️ Splitting documents into chunks...")
                chunks = chunk_documents(
                    documents, 
                    chunk_size=CHUNK_SIZE, 
                    chunk_overlap=CHUNK_OVERLAP
                )
                
                # Step 3: Create vector store with embeddings
                st.info("🧮 Generating embeddings and storing in vector database...")
                vector_store = create_vector_store(chunks)
                
                # Save to session state
                st.session_state.vector_store = vector_store
                st.session_state.documents_processed = True
                
                st.success(
                    f"✅ Successfully processed {len(uploaded_files)} document(s) "
                    f"into {len(chunks)} chunks!"
                )
                
            except ValueError as e:
                st.error(f"❌ Configuration Error: {e}")
            except Exception as e:
                st.error(f"❌ Error processing documents: {e}")
    
    st.markdown("---")
    
    # Show processing status
    if st.session_state.documents_processed:
        st.success("✅ Documents loaded and ready!")
    else:
        st.warning("⚠️ Upload and process documents to start chatting.")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Info section
    st.markdown("---")
    st.markdown("### 💡 How it works")
    st.markdown("""
    1. **Upload** your PDF documents
    2. **Process** them (text extraction → chunking → embedding)
    3. **Ask questions** about your documents
    4. The AI retrieves relevant sections and generates accurate answers
    """)


# --- Main Chat Interface ---
st.title("📚 Archie")
st.caption("Your intelligent document companion powered by Retrieval-Augmented Generation (RAG)")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available (for assistant messages)
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Source Documents"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:** {source['file']}")
                    st.text(source["content"][:300] + "...")
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # Check if documents are processed
    if not st.session_state.documents_processed or st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and process documents first using the sidebar!")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                # Use conversational answer for follow-up context
                answer, source_docs = get_conversational_answer(
                    vector_store=st.session_state.vector_store,
                    query=prompt,
                    chat_history=st.session_state.messages
                )
                
                # Display the answer
                st.markdown(answer)
                
                # Prepare source information for display and storage
                sources = []
                if source_docs:
                    with st.expander("📚 View Source Documents"):
                        for i, doc in enumerate(source_docs, 1):
                            source_name = doc.metadata.get("source", "Unknown")
                            st.markdown(f"**Source {i}:** {source_name}")
                            st.text(doc.page_content[:300] + "...")
                            st.markdown("---")
                            sources.append({
                                "file": source_name,
                                "content": doc.page_content
                            })
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"❌ Error generating answer: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
