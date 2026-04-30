from google import genai

MODEL = "gemini-2.5-flash"

PROMPT_TEMPLATE = """You are an expert ML/AI documentation assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find that in the documentation."
Always cite which source you used.

Context:
{context}

Question: {question}

Answer:"""

def generate(query: str, retrieved_chunks: list[dict], api_key: str) -> dict:
    context = "\n\n".join([
        f"[Source: {c['source']} - {c['url']}]\n{c['content']}"
        for c in retrieved_chunks
    ])

    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    sources = list(set([c["url"] for c in retrieved_chunks]))

    return {
        "answer": response.text,
        "sources": sources
    }

if __name__ == "__main__":
    import os
    from retriever import retrieve
    api_key = os.getenv("GEMINI_API_KEY")
    query = "How do I fine-tune a model with HuggingFace?"
    chunks = retrieve(query)
    result = generate(query, chunks, api_key)
    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(" -", s)