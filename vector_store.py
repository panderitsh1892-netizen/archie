"""
Vector Store Module — FAISS (Facebook AI Similarity Search)
============================================================

This module handles creating and loading the FAISS vector store.

What is a Vector Store?
-----------------------
A vector store is a specialized database designed to store "embeddings" (numerical 
representations of text) and perform fast "similarity search" — finding the stored 
texts that are most similar in meaning to a query.

What is FAISS?
--------------
FAISS (Facebook AI Similarity Search) is an open-source library by Meta for 
efficient similarity search. It's lightweight, fast, and works entirely locally 
without needing a separate server (unlike some alternatives like Pinecone).

How Similarity Search Works:
1. Each text chunk is converted into a vector (list of numbers) called an embedding.
2. These vectors are stored in FAISS.
3. When you ask a question, your question is also converted to a vector.
4. FAISS finds the stored vectors closest to your question vector using distance metrics.
5. The corresponding text chunks are returned as "relevant context."
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import get_google_api_key, EMBEDDING_MODEL, PERSIST_DIRECTORY


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initialize and return the Google Gemini embedding model.
    
    Embeddings convert text into dense numerical vectors that capture 
    semantic meaning. Words/sentences with similar meanings will have 
    vectors that are close together in the high-dimensional vector space.
    """
    api_key = get_google_api_key()
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )


def create_vector_store(chunks: List[Document]) -> FAISS:
    """
    Creates a FAISS vector store from document chunks and saves it locally.
    
    Process:
    1. Takes text chunks (Document objects)
    2. Generates embeddings for each chunk using Gemini
    3. Stores them in a FAISS index
    4. Saves the index to disk for persistence
    
    Args:
        chunks: List of Document objects (text chunks with metadata)
    
    Returns:
        FAISS vector store instance ready for similarity search
    """
    embeddings = get_embeddings()
    
    # Create FAISS vector store from the text chunks
    # This embeds all chunks and builds the search index
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Save the index to disk so we don't have to re-embed next time
    vector_store.save_local(PERSIST_DIRECTORY)
    
    return vector_store


def get_vector_store() -> Optional[FAISS]:
    """
    Loads an existing FAISS vector store from the local directory.
    
    Returns:
        FAISS vector store if it exists, None otherwise
    """
    embeddings = get_embeddings()
    
    if os.path.exists(PERSIST_DIRECTORY):
        return FAISS.load_local(
            PERSIST_DIRECTORY, 
            embeddings,
            allow_dangerous_deserialization=True  # Required for loading pickled data
        )
    return None
