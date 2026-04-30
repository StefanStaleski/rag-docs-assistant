from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from scraper import fetch_docs
import shutil
import os

SOURCES = ["huggingface", "langchain"]
CHROMA_PATH = "../chroma_db"
EMBED_MODEL = "BAAI/bge-small-en"

def run():
    # wipe first, before anything else
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Wiped old ChromaDB")

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    all_chunks = []
    all_metadata = []

    for source in SOURCES:
        docs = fetch_docs(source, max_pages=50)
        for doc in docs:
            chunks = splitter.split_text(doc["content"])
            all_chunks.extend(chunks)
            all_metadata.extend([{"source": source, "url": doc["url"]}] * len(chunks))
        print(f"{source}: {len(all_chunks)} total chunks so far")

    print("Embedding and storing...")
    Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        metadatas=all_metadata,
        persist_directory=CHROMA_PATH
    )
    print(f"Done. {len(all_chunks)} chunks stored.")

if __name__ == "__main__":
    run()