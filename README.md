🩺 Medibot — Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) based medical chatbot that answers healthcare-related questions using a grounded knowledge base built from medical documents.

Instead of relying only on an LLM, Medibot retrieves relevant context from a medical PDF and uses it to generate accurate, context-aware responses.

🚀 Demo

Streamlit-based chatbot interface where users can ask medical questions and get AI-generated answers grounded in medical literature.

📌 Key Features
📄 Upload & process medical PDF documents
🔍 Semantic search using vector embeddings
🧠 Context-aware responses using LLMs
🗂️ FAISS vector database for fast retrieval
💬 Interactive chatbot UI using Streamlit
⚡ Low hallucination via retrieval grounding
🧠 System Architecture
User (Streamlit UI)
        ↓
LangChain Orchestrator
        ↓
Document Ingestion (PDF → Chunks)
        ↓
HuggingFace Embeddings
        ↓
FAISS Vector Store
        ↓
Similarity Search (Top-K Context)
        ↓
LLM (HuggingFace / OpenAI)
        ↓
Generated Medical Answer
        ↓
Streamlit UI Response
🏗️ Tech Stack
Python
LangChain (RAG orchestration)
HuggingFace Transformers
FAISS (Vector Database)
Streamlit (Frontend UI)
PyPDF / PDF loaders
Sentence Transformers (Embeddings)
📂 Project Structure
medibot/
│
├── data/                 # Medical PDF files
├── embeddings/           # Vector store files (FAISS index)
├── src/
│   ├── ingestion.py      # PDF loading & chunking
│   ├── embeddings.py     # Embedding generation
│   ├── vectorstore.py    # FAISS setup
│   ├── retrieval.py      # Similarity search logic
│   ├── llm_chain.py      # Prompt + LLM pipeline
│   └── app.py            # Streamlit UI
│
├── requirements.txt
└── README.md
⚙️ How It Works
1️⃣ Document Ingestion
Medical PDFs are loaded
Text is split into smaller chunks for processing
2️⃣ Embedding Creation
Each chunk is converted into vector embeddings using HuggingFace models
3️⃣ Vector Storage
Embeddings are stored in FAISS for fast similarity search
4️⃣ Query Processing
User question is converted into embedding
FAISS retrieves top relevant chunks
5️⃣ Response Generation
Retrieved context + user query → passed to LLM
Model generates grounded medical response
🧪 Example Use Cases
“What are symptoms of diabetes?”
“Explain hypertension in simple terms”
“What causes chest pain?”
“Side effects of paracetamol?”
🧠 Why RAG Instead of Plain LLM?
Plain LLM	Medibot (RAG)
May hallucinate	Uses real medical text
No grounding	Context-aware answers
Generic responses	Domain-specific responses
⚠️ Disclaimer

Medibot is for educational purposes only and should not be used as a substitute for professional medical advice.

🔧 Setup Instructions
# Clone repo
git clone https://github.com/your-username/medibot.git
cd medibot

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run src/app.py
📈 Future Improvements
Add multi-document support
Improve retrieval with hybrid search (BM25 + embeddings)
Add reranking model for better accuracy
Deploy using Docker + Cloud (AWS / Azure)
Add conversation memory for chat history
