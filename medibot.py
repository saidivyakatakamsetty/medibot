import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="MediBot 💊", page_icon="💊", layout="centered")

st.title("💊 MediBot")
st.caption("Your friendly medical assistant")


@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore/db_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

@st.cache_resource
def load_llm():

    model_id = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3
    )


def get_answer(question, db, llm):

    q = question.lower().strip()

    
    greetings = ["hi", "hello", "hey"]

    if any(g in q for g in greetings):
        return (
            "Hi 😊 I’m MediBot, your friendly medical assistant. "
            "How can I help you today?",
            []
        )

    if "how are you" in q:
        return (
            "I’m doing great 😊 Thanks for asking! "
            "I’m here whenever you need help.",
            []
        )

    if "your name" in q or "who are you" in q:
        return (
            "I’m MediBot 💊, your medical assistant chatbot. "
            "I help explain health-related topics in simple language.",
            []
        )

    
    general_questions = [
        "prime minister", "national bird", "capital", "weather",
        "who is", "what is country", "india", "usa"
    ]

    if any(x in q for x in general_questions):
        return (
            "I’m designed mainly for medical questions 😊 "
            "Please ask me something related to health or medicine.",
            []
        )

    
    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(
        [doc.page_content for doc in docs if doc.page_content]
    )

    # safety fallback
    if len(context.strip()) < 50:
        return (
            "I don’t have enough medical information to answer this clearly.",
            []
        )

    prompt = f"""
You are MediBot, a friendly medical assistant.

Rules:
- Use ONLY the context below
- Be simple and clear
- If unsure, say you don't know

Context:
{context}

Question:
{question}

Answer:
"""

    result = llm(prompt)

    if isinstance(result, list):
        answer = result[0].get("generated_text", result[0].get("text", str(result[0])))
    else:
        answer = str(result)

    return answer, docs
    prompt = f"""
You are MediBot, a friendly medical assistant.

Rules:
- Be simple and clear
- Use ONLY the context below
- If unsure, say you don't have enough information

Context:
{context}

Question:
{question}

Answer:
"""

    result = llm(prompt)

    if isinstance(result, list):
        answer = result[0].get("generated_text", result[0].get("text", str(result[0])))
    else:
        answer = str(result)

    return answer, docs


db = load_vectorstore()
llm = load_llm()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask me anything...")

if user_input:

    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("MediBot is thinking..."):

            answer, sources = get_answer(user_input, db, llm)

            st.markdown(answer)

            # show sources only for medical queries
            if sources:
                with st.expander("View Sources"):
                    for i, doc in enumerate(sources):
                        st.write(f"Document {i+1}")
                        st.write(doc.page_content[:400])
                        st.divider()

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )