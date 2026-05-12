# 🩺 MediBot — Medical RAG Chatbot

MediBot is a **Retrieval-Augmented Generation (RAG)** based medical chatbot that answers health-related questions using context retrieved from a medical PDF knowledge base.

It combines **LangChain + HuggingFace + FAISS** to generate grounded and context-aware responses instead of relying only on an LLM.

---

### 🔄 How it works

1. 📄 Load medical PDF documents
2. ✂️ Split documents into chunks
3. 🔢 Convert chunks into embeddings (HuggingFace)
4. 📦 Store embeddings in FAISS vector database
5. 🔍 User query → semantic similarity search
6. 🧠 Retrieve most relevant chunks
7. 💬 Pass context + query to LLM
8. 📤 Generate grounded medical response

---

# 🛠️ Tech Stack

* Python 🐍
* LangChain 🦜
* HuggingFace Transformers 🤗
* FAISS (Vector Database)
* Streamlit / CLI (UI layer)
* PyPDF / document loaders

---

# 📁 Recommended Project Structure

```
medibot/
│── app.py                     # Streamlit or main UI
│── requirements.txt
│── README.md
│
├── ingestion/
│   ├── load_pdf.py            # Load and preprocess PDF
│   ├── chunking.py            # Text splitting logic
│
├── embeddings/
│   ├── embedder.py            # HuggingFace embeddings
│
├── vectorstore/
│   ├── faiss_store.py         # FAISS index creation & retrieval
│
├── chains/
│   ├── rag_chain.py           # LangChain QA pipeline
│
├── data/
│   ├── medical_book.pdf
│
└── utils/
    ├── helpers.py
```

---

# 🚀 Key Features

* 📚 PDF-based medical knowledge retrieval
* 🔎 Semantic search using embeddings
* 🧠 Context-aware LLM responses
* ⚡ Fast vector search using FAISS
* 💬 Chat-style Q&A interface

---

# ⚠️ Important Disclaimer

MediBot is **not a medical diagnostic tool**. It is intended for educational and informational purposes only. Always consult a certified medical professional for health advice.

---

# 💡 Future Improvements

* 🔁 Add conversation memory (multi-turn chat)
* 📊 Add citation-based answers (source highlighting)
* 🧠 Integrate reranking model for better retrieval
* ☁️ Deploy on AWS / GCP / Azure
* 🔐 Add authentication for users
* 📱 Build mobile-friendly UI

---

# 🧪 Setup Instructions

```bash
# Clone repo
git clone https://github.com/yourusername/medibot.git
cd medibot

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

