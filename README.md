# 📚 Archie — Your Intelligent Document Companion

**Archie** is an enterprise-grade **Retrieval-Augmented Generation (RAG)** assistant that reads, indexes, and converses with your PDF documents, grounding answers in verifiable facts with citations. Built with LangChain, Google Gemini, FAISS, and Streamlit.

## 🧠 What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that enhances LLM responses by grounding them in your own data. Instead of relying solely on the LLM's training data, RAG retrieves relevant information from your documents and feeds it to the LLM as context.

### Why Do We Need RAG?

| Problem | How RAG Solves It |
|---------|------------------|
| LLMs have a **knowledge cutoff** | RAG provides up-to-date information from your docs |
| LLMs **hallucinate** (make up facts) | RAG grounds answers in real document content |
| LLMs don't know your **private data** | RAG lets you query your own PDFs, reports, etc. |
| LLMs have **limited context windows** | RAG retrieves only the most relevant chunks |

## 🏗️ Architecture — The RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
│                                                                 │
│  📄 PDF Upload                                                  │
│       │                                                         │
│       ▼                                                         │
│  📖 Text Extraction (PyPDF2)                                    │
│       │                                                         │
│       ▼                                                         │
│  ✂️  Chunking (RecursiveCharacterTextSplitter)                   │
│       │  - Splits text into ~1000 char chunks                   │
│       │  - 200 char overlap to preserve context                 │
│       ▼                                                         │
│  🧮 Embedding (Google Gemini embedding-001)                     │
│       │  - Converts text chunks into numerical vectors          │
│       │  - Each vector captures the semantic meaning            │
│       ▼                                                         │
│  💾 Vector Storage (ChromaDB)                                   │
│       │  - Stores embeddings locally                            │
│       │  - Enables fast similarity search                       │
│                                                                 │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ Query Time ─ ─ ─ ─ ─ ─ ─ ─ ─             │
│                                                                 │
│  ❓ User Question                                               │
│       │                                                         │
│       ▼                                                         │
│  🔍 Similarity Search (ChromaDB)                                │
│       │  - Converts question to embedding                       │
│       │  - Finds top-k most similar document chunks             │
│       ▼                                                         │
│  📝 Prompt Augmentation                                         │
│       │  - Injects retrieved chunks as "context" in the prompt  │
│       ▼                                                         │
│  🤖 LLM Generation (Gemini 1.5 Flash)                          │
│       │  - Generates answer based on context + question         │
│       ▼                                                         │
│  💬 Answer with Source Citations                                │
└─────────────────────────────────────────────────────────────────┘
```

## 🔑 Key Concepts Explained

### Chunking
Large documents can't fit in an LLM's context window. **Chunking** splits documents into smaller pieces (~1000 characters). We use **overlap** (200 chars) so that information at chunk boundaries isn't lost.

### Embeddings
An **embedding** is a list of numbers (vector) that represents the meaning of text. Similar texts have similar vectors. For example, "dog" and "puppy" would have vectors close together.

### Vector Database (ChromaDB)
A specialized database for storing and searching embeddings. When you ask a question, it converts your question to a vector and finds the most similar stored vectors — this is called **similarity search**.

### Similarity Search
Uses **cosine similarity** to measure the angle between two vectors. Smaller angle = more similar meaning. This is how we find which document chunks are most relevant to your question.

## 📂 Project Structure

```
archie/
├── app.py                 # Streamlit UI — main entry point
├── rag_engine.py          # Core RAG logic — retrieval + generation
├── document_processor.py  # PDF loading + text chunking
├── vector_store.py        # FAISS vector store operations
├── config.py              # Configuration and constants
├── requirements.txt       # Python dependencies
├── .env.example           # API key template
└── sample_docs/           # Place your PDFs here
```

## 🚀 Setup & Installation

### 1. Get a Google Gemini API Key (Free)
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key

### 2. Install Dependencies
```bash
cd archie
pip install -r requirements.txt
```

### 3. Configure API Key
```bash
cp .env.example .env
# Edit .env and paste your API key:
# GOOGLE_API_KEY=your_actual_key_here
```

### 4. Run the App
```bash
streamlit run app.py
```

### 5. Use the App
1. Upload one or more PDF files via the sidebar
2. Click "Process Documents" and wait for processing
3. Start asking questions in the chat!

## 🎯 Example Questions
After uploading a document, try:
- "What is the main topic of this document?"
- "Summarize the key points"
- "What does the author say about [specific topic]?"
- Follow-up: "Can you explain that in simpler terms?"

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Google Gemini 1.5 Flash | Answer generation |
| Embeddings | Google embedding-001 | Text → Vector conversion |
| Vector Store | ChromaDB | Embedding storage & search |
| Framework | LangChain | Orchestration & chaining |
| PDF Processing | PyPDF2 | Extract text from PDFs |
| UI | Streamlit | Web interface |

## 📚 Learning Resources
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Google Gemini API Docs](https://ai.google.dev/docs)
- [What are Embeddings? (OpenAI)](https://platform.openai.com/docs/guides/embeddings)
