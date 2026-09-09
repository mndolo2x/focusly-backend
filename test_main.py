import os
import io
import jwt
import pytest
from fastapi.testclient import TestClient
from main import app
from services.ollama_service import ollama_service
from services.review_service import review_service
from services.tts_service import tts_service

client = TestClient(app)

def get_auth_headers():
    token = jwt.encode(
        {"sub": "user_123", "email": "test@focusly.ai", "user_metadata": {"grade_level": "11th", "exam_targets": ["SAT"]}},
        "test_secret",
        algorithm="HS256"
    )
    return {"Authorization": f"Bearer {token}"}

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

def test_tts_health_endpoint():
    response = client.get("/api/health/tts")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "mp3_export_supported" in data

@pytest.mark.anyio
async def test_tts_generate_audio_mp3_export():
    out_path = "/tmp/test_narration.mp3"
    result_path = await tts_service.generate_audio("Hello, welcome to Focusly lessons.", out_path)
    assert os.path.exists(result_path)
    assert result_path.endswith(".mp3")
    if os.path.exists(result_path):
        os.remove(result_path)

def test_unauthenticated_docs_access():
    response = client.get("/api/documents")
    assert response.status_code == 401

def test_admin_exam_profiles():
    response = client.get("/admin/exam-profiles")
    assert response.status_code == 200
    profiles = response.json()
    assert isinstance(profiles, list)

def test_migration_sql_schema_exists():
    sql_path = os.path.join(os.path.dirname(__file__), "migrations", "001_initial_schema.sql")
    assert os.path.exists(sql_path)

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
