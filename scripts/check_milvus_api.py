"""Real HTTP upload -> retrieval -> delete check against a real Milvus server."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import httpx
from pymilvus import MilvusClient


def main():
    root = Path(__file__).resolve().parents[1]
    uri = os.getenv("MILVUS_TEST_URI", "http://127.0.0.1:19530")
    token = os.getenv("MILVUS_TEST_TOKEN", "")
    collection = "api_check_" + uuid4().hex
    api_token = uuid4().hex
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="milvus-api-") as directory:
        env = {**os.environ, "RAG_BACKEND": "milvus", "MILVUS_URI": uri,
               "MILVUS_TOKEN": token, "MILVUS_COLLECTION": collection,
               "MILVUS_TIMEOUT_SECONDS": "30", "MILVUS_EMBEDDING_REVISION": "api-hash-v1",
               "LLM_PROVIDER": "fake", "EMBEDDING_PROVIDER": "hash",
               "EMBEDDING_DIMENSIONS": "32", "RAG_VECTOR_DIMENSIONS": "32",
               "RERANK_PROVIDER": "none", "API_TOKEN": api_token,
               "SQLITE_URL": "sqlite:///" + str(Path(directory) / "sessions.db")}
        with open(Path(directory)/"api.log", "w", encoding="utf-8") as output:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:create_app",
                                        "--factory", "--host", "127.0.0.1", "--port", str(port)],
                                       cwd=root, env=env, stdout=output, stderr=output)
            admin = MilvusClient(uri=uri, token=token)
            try:
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60,
                                  headers={"Authorization": "Bearer " + api_token},
                                  trust_env=False) as client:
                    for _ in range(60):
                        if process.poll() is not None:
                            raise RuntimeError("API process terminated before readiness")
                        try:
                            response = client.get("/health/ready")
                            if response.status_code == 200:
                                break
                        except httpx.ConnectError:
                            pass
                        time.sleep(1)
                    else:
                        raise RuntimeError("API readiness timeout")
                    assert response.json()["rag"]["backend"] == "milvus"
                    files = sorted((root/"examples/documents").glob("*.pdf"))
                    assert files, "Missing synthetic PDF fixture"
                    with files[0].open("rb") as pdf:
                        response = client.post("/api/v1/documents/upload",
                                               files={"file": (files[0].name, pdf, "application/pdf")})
                    response.raise_for_status()
                    doc = response.json()["document"]
                    response = client.get("/api/v1/documents")
                    response.raise_for_status()
                    assert doc in response.json()["documents"]
                    response = client.post("/api/v1/chat", json={"message": "差旅住宿报销标准是什么？",
                                                                "user_id": "api-check"})
                    response.raise_for_status()
                    assert any(s["doc_id"] == doc["doc_id"] for s in response.json()["sources"])
                    response = client.delete("/api/v1/documents/" + doc["doc_id"])
                    response.raise_for_status()
                    assert response.json()["deleted"]
                    assert client.get("/api/v1/documents").json()["documents"] == []
                    result = {"backend": "milvus", "http_pdf_upload": True,
                              "chat_source_verified": True, "delete_verified": True,
                              "chunk_count": doc["chunk_count"], "llm": "fake",
                              "embedding": "hash", "not_a_quality_benchmark": True}
                    (root/"milvus-api-result.json").write_text(
                        json.dumps(result, indent=2), encoding="utf-8")
                    print(json.dumps(result))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                if admin.has_collection(collection_name=collection):
                    admin.drop_collection(collection_name=collection)
                admin.close()


if __name__ == "__main__":
    main()
