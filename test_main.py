import os
import io
import jwt
import pytest
from datetime import datetime, timedelta
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

def test_ai_study_plan_builder_flow(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    exam_date_str = (datetime.utcnow() + timedelta(days=14)).isoformat() + "Z"

    # 1. Create AI Study Plan (POST /api/study-plan)
    post_res = client.post(
        "/api/study-plan",
        json={"exam_date": exam_date_str, "subject": "AP Chemistry", "documents": []},
        headers=get_auth_headers()
    )
    assert post_res.status_code == 200
    plan = post_res.json()
    assert "schedule" in plan
    assert plan["subject"] == "AP Chemistry"

    # 2. Get Current Plan (GET /api/study-plan)
    get_res = client.get("/api/study-plan", headers=get_auth_headers())
    assert get_res.status_code == 200
    assert get_res.json()["id"] == plan["id"]

    # 3. Get Today's Tasks (GET /api/study-plan/today)
    today_res = client.get("/api/study-plan/today", headers=get_auth_headers())
    assert today_res.status_code == 200
    today_data = today_res.json()
    assert "tasks" in today_data
    assert len(today_data["tasks"]) >= 1

    task_id = today_data["tasks"][0].get("id", "task_1_1")

    # 4. Complete Task (POST /api/study-plan/tasks/{id}/complete)
    comp_res = client.post(f"/api/study-plan/tasks/{task_id}/complete", headers=get_auth_headers())
    assert comp_res.status_code == 200
    assert comp_res.json()["completed"] is True

    # 5. Update/Regenerate Plan (PUT /api/study-plan)
    put_res = client.put(
        "/api/study-plan",
        json={"subject": "AP Chemistry Advanced"},
        headers=get_auth_headers()
    )
    assert put_res.status_code == 200
    assert put_res.json()["subject"] == "AP Chemistry Advanced"

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
