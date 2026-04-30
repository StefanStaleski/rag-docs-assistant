import mlflow
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "mlflow", "mlruns.db")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{DB_PATH}")
EXPERIMENT_NAME = "rag-docs-assistant"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def track_query(question: str, answer: str, sources: list[str], chunks: list[dict], latency_ms: float):
    with mlflow.start_run():
        # params
        mlflow.log_param("question", question[:200])
        mlflow.log_param("model", "gemini-2.0-flash")
        mlflow.log_param("embed_model", "BAAI/bge-small-en")
        mlflow.log_param("num_chunks_retrieved", len(chunks))

        # metrics
        mlflow.log_metric("latency_ms", latency_ms)
        mlflow.log_metric("num_sources", len(set(sources)))
        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_metric("answered", 0 if "couldn't find" in answer.lower() else 1)

        # artifacts
        mlflow.log_text(answer, "answer.txt")
        mlflow.log_text("\n".join(sources), "sources.txt")