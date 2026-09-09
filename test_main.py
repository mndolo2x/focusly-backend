import os
import io
import jwt
import pytest
from fastapi.testclient import TestClient
from main import app
from services.ollama_service import ollama_service

client = TestClient(app)

def get_auth_headers(is_admin: bool = False):
    token = jwt.encode(
        {"sub": "user_123", "email": "test@focusly.ai", "user_metadata": {"is_admin": is_admin, "grade_level": "11th"}},
        "test_secret",
        algorithm="HS256"
    )
    return {"Authorization": f"Bearer {token}"}

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_dashboard_progress_metrics_and_caching(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    # 1. First GET request - calculates & caches dashboard metrics
    res1 = client.get("/api/dashboard/progress", headers=get_auth_headers())
    assert res1.status_code == 200
    data1 = res1.json()

    required_keys = [
        "total_lessons", "completed_lessons", "average_quiz_score",
        "quiz_score_history", "weak_areas", "reviews_due_today",
        "upcoming_reviews", "study_plan", "exam_countdown", "cached_at"
    ]
    for key in required_keys:
        assert key in data1

    assert isinstance(data1["total_lessons"], int)
    assert isinstance(data1["completed_lessons"], int)
    assert isinstance(data1["quiz_score_history"], list)
    assert isinstance(data1["weak_areas"], list)
    assert isinstance(data1["upcoming_reviews"], list)

    # 2. Second GET request - returns cached response (matching cached_at timestamp)
    res2 = client.get("/api/dashboard/progress", headers=get_auth_headers())
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cached_at"] == data1["cached_at"]

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
