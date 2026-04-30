from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "../chroma_db"
EMBED_MODEL = "BAAI/bge-small-en"

def load_retriever(k: int = 5):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": k})

def retrieve(query: str, k: int = 5) -> list[dict]:
    retriever = load_retriever(k)
    docs = retriever.invoke(query)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "url": doc.metadata.get("url")
        }
        for doc in docs
    ]

if __name__ == "__main__":
    query = "How do I fine-tune a model with HuggingFace?"
    results = retrieve(query)
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} [{r['source']}] ---")
        print(r['url'])
        print(r['content'][:200])