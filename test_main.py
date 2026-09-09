import os
import jwt
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

def test_migration_sql_schema_exists():
    sql_path = os.path.join(os.path.dirname(__file__), "migrations", "001_initial_schema.sql")
    assert os.path.exists(sql_path)
    with open(sql_path, "r", encoding="utf-8") as f:
        content = f.read()

    expected_tables = [
        "public.documents", "public.summaries", "public.quiz_questions",
        "public.quiz_attempts", "public.review_items", "public.study_plans",
        "public.exam_profiles", "public.user_usage", "public.shared_links"
    ]
    for tbl in expected_tables:
        assert tbl in content

    expected_indexes = [
        "idx_documents_user_id", "idx_documents_status",
        "idx_quiz_attempts_user_id", "idx_review_items_user_id",
        "idx_review_items_next_review_date"
    ]
    for idx in expected_indexes:
        assert idx in content

    assert "get_user_usage" in content
    assert "handle_new_user_usage" in content
    assert "ROW LEVEL SECURITY" in content

def test_auth_me_with_jwt_token(monkeypatch):
    # Test protected /api/auth/me with mock JWT token
    token = jwt.encode(
        {"sub": "user_123", "email": "test@focusly.ai", "user_metadata": {"grade_level": "11th", "exam_targets": ["SAT"]}},
        "test_secret",
        algorithm="HS256"
    )
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "user_123"
    assert data["email"] == "test@focusly.ai"
    assert data["grade_level"] == "11th"
    assert data["exam_targets"] == ["SAT"]

def test_auth_logout_with_jwt_token(monkeypatch):
    token = jwt.encode(
        {"sub": "user_123", "email": "test@focusly.ai"},
        "test_secret",
        algorithm="HS256"
    )
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    response = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json() == {"message": "Successfully logged out"}

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)

@pytest.mark.anyio
async def test_ollama_service_embed_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.embed("Test text", max_retries=2, retry_delay=0.01)
    assert "Ollama embedding connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
