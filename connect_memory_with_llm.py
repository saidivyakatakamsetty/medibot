import os
import sys
import requests
from typing import Optional, List

from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# ── 1. Auth ───────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Set it as an environment variable.")

# ── 2. Constants ──────────────────────────────────────────────────────────────
HUGGINGFACE_REPO_ID = "facebook/bart-large-cnn"
EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
DB_FAISS_PATH       = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Don't provide anything outside of the given context.

Context: {context}
Question: {input}

Start the answer directly, no small talk please.
"""

# ── 3. Custom LLM (direct API call — no version conflicts) ────────────────────
class HuggingFaceLLM(LLM):
    repo_id: str
    hf_token: str
    max_new_tokens: int = 512
    temperature: float = 0.5

    @property
    def _llm_type(self) -> str:
        return "huggingface"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }
        }
        url = f"https://router.huggingface.co/hf-inference/models/{self.repo_id}"

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "Request timed out. The model may be loading, please try again."
        except requests.exceptions.HTTPError as e:
            return f"API error: {e}"

        result = response.json()

        # Handle model loading state
        if isinstance(result, dict) and "error" in result:
            return f"Model error: {result['error']}"

        # Handle different response formats
        if isinstance(result, list):
            if "generated_text" in result[0]:
                return result[0]["generated_text"]
            elif "summary_text" in result[0]:
                return result[0]["summary_text"]

        return str(result)

# ── 4. Load LLM ───────────────────────────────────────────────────────────────
def load_llm(repo_id: str, hf_token: str) -> HuggingFaceLLM:
    return HuggingFaceLLM(repo_id=repo_id, hf_token=hf_token)

# ── 5. Prompt ─────────────────────────────────────────────────────────────────
def build_prompt(template: str) -> PromptTemplate:
    return PromptTemplate(
        template=template,
        input_variables=["context", "input"],
    )

# ── 6. Vector store ───────────────────────────────────────────────────────────
def load_vectorstore(db_path: str, hf_token: str) -> FAISS:
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{db_path}'. "
            "Run create_memory_for_llm.py first."
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"token": hf_token},
    )
    return FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

# ── 7. Build chain ────────────────────────────────────────────────────────────
def build_chain(llm, vectorstore: FAISS):
    prompt = build_prompt(CUSTOM_PROMPT_TEMPLATE)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return create_retrieval_chain(retriever, combine_docs_chain)

# ── 8. Query helper ───────────────────────────────────────────────────────────
def run_query(chain, user_query: str) -> None:
    if not user_query.strip():
        print("Query cannot be empty.")
        return
    try:
        response = chain.invoke({"input": user_query})
    except Exception as e:
        import traceback
        print(f"Error during query: {e}")
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("RESULT:")
    answer = (
        response.get("answer")
        or response.get("output")
        or response.get("text")
        or "No answer found."
    )

    # ── Clean up prompt leaking into answer ───────────────────
    cleanup_phrases = [
        "Use the pieces of information provided in the context to answer the user's question.",
        "Don't make up an answer.",
        "Don't provide anything outside of the given context.",
        "Start the answer directly, no small talk please.",
    ]
    for phrase in cleanup_phrases:
        answer = answer.replace(phrase, "").strip()

    print(answer)
    print("\nSOURCE DOCUMENTS:")
    for i, doc in enumerate(response.get("context", []), 1):
        source = doc.metadata.get("source", "Unknown")
        print(f"  [{i}] {source}\n      {doc.page_content[:200].strip()}...")
    print("=" * 60)
# ── 9. Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading vector store...")
    try:
        vectorstore = load_vectorstore(DB_FAISS_PATH, HF_TOKEN)
    except FileNotFoundError as e:
        print(f"Startup error: {e}")
        sys.exit(1)

    print("Loading LLM...")
    llm   = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
    chain = build_chain(llm, vectorstore)

    print("\nRAG assistant ready. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        run_query(chain, user_query)

if __name__ == "__main__":
    main()