"""
RAG Engine - The Core of the RAG Chatbot
=========================================

This module ties together the entire RAG (Retrieval-Augmented Generation) pipeline:

1. RETRIEVAL: When a user asks a question, we search the vector store for the most
   relevant document chunks using similarity search (comparing embeddings).

2. AUGMENTATION: We take the retrieved chunks and inject them into the LLM's prompt
   as "context". This gives the LLM specific knowledge it wouldn't otherwise have.

3. GENERATION: The LLM generates an answer based on the provided context + question.

Why RAG?
--------
- LLMs have a knowledge cutoff date — they don't know about your private documents.
- LLMs can "hallucinate" — make up facts that sound correct but aren't.
- RAG grounds the LLM's responses in your actual data, making them accurate and verifiable.
"""

from typing import List, Dict, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from config import get_google_api_key, LLM_MODEL


def extract_text(content) -> str:
    """
    Extract plain text from LLM response content.
    Newer Gemini API versions may return content as a list of content blocks
    (e.g., [{'type': 'text', 'text': '...', 'extras': {...}}]) instead of a plain string.
    This helper handles both formats.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Extract text from each content block and join them
        parts = []
        for block in content:
            if isinstance(block, dict) and 'text' in block:
                parts.append(block['text'])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


# --- Prompt Template ---
# This is the prompt we send to the LLM. It includes:
# 1. A system instruction telling the LLM how to behave
# 2. The retrieved context (relevant chunks from our documents)
# 3. The user's question
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on 
the provided context. Follow these rules strictly:

1. ONLY use information from the provided context to answer the question.
2. If the context doesn't contain enough information to answer, say "I don't have enough 
   information in the provided documents to answer this question."
3. When answering, cite which part of the context you're referring to.
4. Be concise but thorough in your answers.

Context:
{context}

Question: {question}

Answer:"""


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize and return the Google Gemini LLM.
    
    ChatGoogleGenerativeAI is LangChain's wrapper around Google's Gemini API.
    It allows us to use Gemini just like any other LLM in the LangChain ecosystem.
    
    Parameters we set:
    - model: Which Gemini model to use (gemini-1.5-flash is fast and free-tier friendly)
    - temperature: Controls randomness (0 = deterministic, 1 = creative). 
      We use 0.3 for factual answers with slight flexibility.
    """
    api_key = get_google_api_key()
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0.3,
    )
    return llm


def retrieve_relevant_chunks(
    vector_store: FAISS, 
    query: str, 
    k: int = 4
) -> List[Document]:
    """
    Retrieve the most relevant document chunks for a given query.
    
    How Similarity Search Works:
    1. The query text is converted into an embedding (a vector of numbers).
    2. This query vector is compared against all stored document vectors in FAISSDB.
    3. The comparison uses "cosine similarity" — measuring the angle between vectors.
       - Vectors pointing in similar directions = semantically similar text.
    4. The top-k most similar chunks are returned.
    
    Args:
        vector_store: The FAISSDB vector store containing our document embeddings.
        query: The user's question.
        k: Number of top results to retrieve (default: 4).
           More chunks = more context but potentially more noise.
    
    Returns:
        List of the most relevant Document chunks.
    """
    # similarity_search converts the query to an embedding and finds nearest neighbors
    relevant_docs = vector_store.similarity_search(query, k=k)
    return relevant_docs


def format_context(documents: List[Document]) -> str:
    """
    Format retrieved documents into a single context string for the LLM.
    
    We include the source metadata so the LLM can cite which document
    the information came from. This adds transparency and trustworthiness.
    """
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(
            f"[Source {i}: {source}]\n{doc.page_content}\n"
        )
    return "\n---\n".join(context_parts)


def generate_answer(
    vector_store: FAISS, 
    query: str, 
    chat_history: List[Dict[str, str]] = None
) -> Tuple[str, List[Document]]:
    """
    The main RAG pipeline function — ties everything together.
    
    The RAG Pipeline:
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
    │  User Query  │ --> │  Retrieval   │ --> │  Augment     │ --> │  Generation  │
    │  "What is X?"│     │  (Vector DB) │     │  (Add context│     │  (LLM Answer)│
    └─────────────┘     └──────────────┘     │  to prompt)  │     └──────────────┘
                                              └─────────────┘
    
    Args:
        vector_store: The FAISSDB vector store with document embeddings.
        query: The user's question.
        chat_history: Previous conversation messages (for context continuity).
    
    Returns:
        Tuple of (answer_text, source_documents) so we can show citations.
    """
    # Step 1: RETRIEVE — Find relevant document chunks
    relevant_docs = retrieve_relevant_chunks(vector_store, query)
    
    if not relevant_docs:
        return (
            "No relevant information found in the uploaded documents. "
            "Please try rephrasing your question.",
            []
        )
    
    # Step 2: AUGMENT — Format the context and build the prompt
    context = format_context(relevant_docs)
    
    # Create the prompt using LangChain's ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # Step 3: GENERATE — Send the augmented prompt to the LLM
    llm = get_llm()
    
    # Create the chain: prompt -> LLM
    # LangChain's pipe operator (|) chains components together
    chain = prompt | llm
    
    # Invoke the chain with our context and question
    response = chain.invoke({
        "context": context,
        "question": query
    })
    
    # response.content contains the LLM's text answer
    return extract_text(response.content), relevant_docs


def get_conversational_answer(
    vector_store: FAISS,
    query: str,
    chat_history: List[Dict[str, str]]
) -> Tuple[str, List[Document]]:
    """
    Enhanced version that considers chat history for follow-up questions.
    
    Why chat history matters:
    If a user asks "What is RAG?" and then asks "How does it work?", the second
    question needs context from the first — "it" refers to RAG. We reformulate
    the question to be self-contained before doing retrieval.
    """
    if chat_history and len(chat_history) > 0:
        # Build context from recent chat history
        history_text = "\n".join(
            f"{msg['role']}: {msg['content']}" 
            for msg in chat_history[-4:]  # Last 4 messages for context
        )
        
        # Reformulate the question to be self-contained
        reformulation_prompt = ChatPromptTemplate.from_template(
            """Given the following conversation history and a follow-up question, 
            reformulate the follow-up question to be a standalone question that 
            captures the full context.

            Chat History:
            {history}

            Follow-up Question: {question}

            Standalone Question:"""
        )
        
        llm = get_llm()
        chain = reformulation_prompt | llm
        reformulated = chain.invoke({
            "history": history_text,
            "question": query
        })
        
        # Use the reformulated question for retrieval
        return generate_answer(vector_store, extract_text(reformulated.content), chat_history)
    
    # No history — just do normal RAG
    return generate_answer(vector_store, query, chat_history)
