# Medibot 🩺

Medibot is a medical question-answering chatbot built using a Retrieval-Augmented Generation (RAG) pipeline.  
Instead of relying only on an LLM, it retrieves relevant context from a medical textbook (PDF) and uses it to generate more grounded responses.

---

## What this project does

- Takes a medical PDF as the knowledge source  
- Splits it into smaller text chunks  
- Converts chunks into embeddings  
- Stores them in a FAISS vector database  
- Retrieves relevant context based on user queries  
- Uses an LLM to generate final responses using that context  

---

## Tech stack

- Python  
- LangChain  
- HuggingFace Transformers  
- FAISS (vector database)  
- PyPDF  

---

## Project structure
medical_chatbot/
│
├── medibot.py
├── create_memory_for_llm.py
├── connect_memory_with_llm.py
├── requirements.txt
│
├── data/
│ └── Medical PDF (GALE Encyclopedia of Medicine)
│
├── vectorstore/
│ └── FAISS index files

---

## How to run

### 1. Clone the repo
```bash
git clone https://github.com/saidivyakatakamsetty/medibot.git
cd medibot
2. Create virtual environment
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Build vector database
python create_memory_for_llm.py

5. Run chatbot
python medibot.py

What I learned from this project
How RAG pipelines work end-to-end
Working with embeddings and vector search (FAISS)
Connecting external knowledge sources with LLMs
Structuring an LLM-based application in Python
