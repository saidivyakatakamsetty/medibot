import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── 1. Constants ──────────────────────────────────────────────────────────────
DATA_PATH     = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # FIX: correct casing (was all-miniLM-L6-V2)

# ── 2. Load PDFs ──────────────────────────────────────────────────────────────
def load_pdf_files(data: str):
    """Load all PDF files from the given directory."""
    if not os.path.exists(data):
        raise FileNotFoundError(
            f"Data directory '{data}' not found. "
            "Create it and add your PDF files inside."
        )

    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    if not documents:
        raise ValueError(f"No PDF files found in '{data}'. Add PDFs and try again.")

    return documents

# ── 3. Split into chunks ──────────────────────────────────────────────────────
def create_chunks(extracted_data):
    """Split documents into overlapping chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    return text_splitter.split_documents(extracted_data)

# ── 4. Embedding model ────────────────────────────────────────────────────────
def get_embedding_model():
    """Load the HuggingFace sentence embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ── 5. Build & save FAISS index ───────────────────────────────────────────────
def build_vectorstore(text_chunks, embedding_model, db_path: str):
    """Create FAISS vector store from chunks and save to disk."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)  # FIX: create folder if missing
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(db_path)
    return db

# ── 6. Main ───────────────────────────────────────────────────────────────────
def main():
    print("Step 1: Loading PDF files...")
    try:
        documents = load_pdf_files(DATA_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return
    print(f"  Loaded {len(documents)} page(s).")

    print("Step 2: Splitting into chunks...")
    text_chunks = create_chunks(documents)
    print(f"  Created {len(text_chunks)} chunk(s).")

    print("Step 3: Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Step 4: Building and saving FAISS vector store...")
    build_vectorstore(text_chunks, embedding_model, DB_FAISS_PATH)
    print(f"  Vector store saved to '{DB_FAISS_PATH}'.")

    print("\nDone! You can now run connect_memory_with_llm.py")

if __name__ == "__main__":
    main()