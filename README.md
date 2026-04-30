# RAG Docs Assistant

A retrieval-augmented generation (RAG) system that answers questions about ML/AI documentation with cited sources. Ask anything about HuggingFace Transformers or LangChain and get grounded, accurate answers backed by the actual documentation.

---

## What it does

1. Scrapes and indexes HuggingFace and LangChain documentation
2. Embeds each chunk using `BAAI/bge-small-en` and stores vectors in ChromaDB
3. At query time, embeds the user's question and retrieves the most semantically similar chunks
4. Passes retrieved context to Gemini 2.0 Flash to generate a grounded answer with citations
5. Tracks every query (latency, answer length, retrieval count, answered/unanswered) in MLFlow

---

## Architecture

```
User Question
     │
     ▼
Embed Question (BAAI/bge-small-en)
     │
     ▼
Vector Search → ChromaDB → Top-K Chunks
     │
     ▼
Prompt + Context → Gemini 2.0 Flash
     │
     ▼
Answer + Sources → FastAPI → Streamlit UI
     │
     ▼
MLFlow Experiment Tracker
```

---

## Stack

| Layer | Technology |
|---|---|
| Embedding model | `BAAI/bge-small-en` via HuggingFace |
| Vector store | ChromaDB |
| Orchestration | LangChain |
| LLM | Gemini 2.0 Flash (Google AI) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Experiment tracking | MLFlow |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
rag-docs-assistant/
├── ingestion/
│   ├── scraper.py        # crawl and clean doc pages
│   └── pipeline.py       # chunk, embed, store in ChromaDB
├── retrieval/
│   ├── retriever.py      # semantic search over ChromaDB
│   └── generator.py      # prompt construction + Gemini call
├── api/
│   └── main.py           # FastAPI endpoint
├── ui/
│   └── app.py            # Streamlit chat interface
├── mlflow/
│   └── tracker.py        # MLFlow experiment logging
├── docker/
│   └── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Getting started

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- Gemini API key (get one free at [aistudio.google.com](https://aistudio.google.com))

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/rag-docs-assistant.git
cd rag-docs-assistant
```

### 2. Set up environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Set your API key

```bash
export GEMINI_API_KEY=your_key_here
```

### 4. Run the ingestion pipeline

This scrapes the docs, chunks them, embeds them, and stores them in ChromaDB.

```bash
cd ingestion && python pipeline.py
```

Expected output: ~1000+ chunks stored across HuggingFace and LangChain docs.

### 5. Start the API

```bash
cd api && uvicorn main:app --reload
```

### 6. Start the UI

```bash
cd ui && streamlit run app.py
```

Open `http://localhost:8501` and start asking questions.

---

## Running with Docker

```bash
cp .env.example .env
# add your GEMINI_API_KEY to .env

docker compose up --build
```

- UI: `http://localhost:8501`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

---

## MLFlow tracking

Every query logs:
- `latency_ms` — end-to-end response time
- `answered` — 1 if answer was found, 0 if not
- `answer_length` — character count of the response
- `num_sources` — number of unique sources retrieved
- `question`, `model`, `embed_model` — run parameters

Start the MLFlow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlruns.db
```

Open `http://localhost:5000` → Evaluation runs.

---

## API reference

### `POST /query`

```json
{
  "question": "How do I fine-tune a model with HuggingFace?",
  "k": 5
}
```

Response:

```json
{
  "answer": "You can fine-tune a model for various tasks...",
  "sources": [
    "https://huggingface.co/docs/transformers/training",
    "https://huggingface.co/docs/transformers/notebooks"
  ]
}
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## Data sources

- [HuggingFace Transformers documentation](https://huggingface.co/docs/transformers)
- [LangChain documentation](https://python.langchain.com/docs)