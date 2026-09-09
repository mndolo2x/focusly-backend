import os
import io
import jwt
import pytest
from fastapi.testclient import TestClient
from main import app
from services.ollama_service import ollama_service
from services.review_service import review_service

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
    assert "available_models" in data

def test_unauthenticated_docs_access():
    response = client.get("/api/documents")
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

def test_sm2_algorithm_math():
    # Quality = 5 (perfect response)
    ef, interval, reps, next_date = review_service.calculate_next_review(ease_factor=2.5, interval=1, quality=5, repetitions=1)
    assert ef > 2.5
    assert reps == 2
    assert interval == 6

    # Quality = 2 (complete blackout/fail) -> resets
    ef_fail, interval_fail, reps_fail, _ = review_service.calculate_next_review(ease_factor=ef, interval=6, quality=2, repetitions=2)
    assert reps_fail == 0
    assert interval_fail == 1

def test_reviews_api_flow(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    # Schedule a test review item
    import asyncio
    item = asyncio.run(review_service.schedule_review("user_123", "q_100", quality=1))
    item_id = item["id"]

    # GET /api/reviews/today
    res_today = client.get("/api/reviews/today", headers=get_auth_headers())
    assert res_today.status_code == 200
    data_today = res_today.json()
    assert "due_count" in data_today
    assert "items" in data_today

    # POST /api/reviews/submit
    res_submit = client.post(
        "/api/reviews/submit",
        json={"review_item_id": item_id, "quality": 4},
        headers=get_auth_headers()
    )
    assert res_submit.status_code == 200
    sub_data = res_submit.json()
    assert sub_data["interval"] >= 1
    assert "next_review_date" in sub_data

    # GET /api/reviews/upcoming
    res_upcoming = client.get("/api/reviews/upcoming?days=7", headers=get_auth_headers())
    assert res_upcoming.status_code == 200
    up_data = res_upcoming.json()
    assert "total_upcoming" in up_data
    assert "schedule" in up_data

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
