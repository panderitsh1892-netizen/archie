import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration Constants
# The LLM model to use for generating answers
LLM_MODEL = "gemini-flash-lite-latest"
# The embedding model to use for converting text to vector representations
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Document Processing Config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Directory to persist the FAISS vector store locally
PERSIST_DIRECTORY = "./faiss_index"

def get_google_api_key():
    """Retrieves the Google API Key from the environment."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in the .env file.")
    return api_key
