import pytest
from fastapi.testclient import TestClient
from main import app
from services.ollama_service import ollama_service

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "message" in json_data
    assert "disclaimer" in json_data

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "environment": "development"}

def test_ollama_health_endpoint():
    response = client.get("/api/health/ollama")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "base_url" in data
    assert "available_models" in data

def test_unauthenticated_docs_access():
    response = client.get("/documents")
    assert response.status_code == 401

def test_admin_exam_profiles():
    response = client.get("/admin/exam-profiles")
    assert response.status_code == 200
    profiles = response.json()
    assert isinstance(profiles, list)
    assert len(profiles) >= 1

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    # Test error handling and retry mechanism when server is unavailable
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)

@pytest.mark.anyio
async def test_ollama_service_embed_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.embed("Test text", max_retries=2, retry_delay=0.01)
    assert "Ollama embedding connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
