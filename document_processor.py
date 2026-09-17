import PyPDF2
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def process_pdfs(pdf_files) -> List[Document]:
    """
    Extracts text from uploaded PDF files and converts them into LangChain Document objects.
    """
    documents = []
    for pdf_file in pdf_files:
        # PyPDF2 PdfReader parses the uploaded file object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        # Iterate through all the pages and extract text
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
            
        # Create a Document object with metadata (useful for citations)
        if text:
            documents.append(Document(page_content=text, metadata={"source": pdf_file.name}))
            
    return documents

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits large documents into smaller chunks.
    
    Why chunking?
    LLMs have a context window limit (how much text they can process at once). 
    If a PDF is 100 pages, we can't pass the whole thing to the LLM. 
    Chunking breaks it into smaller pieces.
    
    Why overlap?
    If we cut text exactly at 1000 characters, we might split a sentence or thought in half. 
    Overlap ensures that the end of one chunk is repeated at the beginning of the next, 
    preserving context across chunk boundaries.
    """
    # RecursiveCharacterTextSplitter tries to split on paragraphs, then sentences, then words, 
    # ensuring that chunks are as semantically cohesive as possible.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    return chunks
