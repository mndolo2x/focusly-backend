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

def test_admin_research_unauthorized_for_non_admin(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")
    response = client.post(
        "/api/admin/exams/research",
        json={"exam_name": "Cambridge IGCSE Biology"},
        headers=get_auth_headers(is_admin=False)
    )
    assert response.status_code == 403
    assert "Admin privileges required" in response.json()["detail"]

def test_admin_exam_research_and_crud_flow(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    # 1. Research Exam with Admin Auth
    res_post = client.post(
        "/api/admin/exams/research",
        json={"exam_name": "Cambridge IGCSE Biology"},
        headers=get_auth_headers(is_admin=True)
    )
    assert res_post.status_code == 200
    data = res_post.json()
    assert "cambridge" in data["id"] or "biology" in data["id"]
    assert data["exam_name"] is not None
    assert "time_limit" in data
    assert "disclaimer_text" in data

    exam_id = data["id"]

    # 2. List Exams
    res_list = client.get("/api/admin/exams", headers=get_auth_headers(is_admin=True))
    assert res_list.status_code == 200
    assert isinstance(res_list.json(), list)

    # 3. Get Specific Exam
    res_get = client.get(f"/api/admin/exams/{exam_id}", headers=get_auth_headers(is_admin=True))
    assert res_get.status_code == 200
    assert res_get.json()["id"] == exam_id

    # 4. Manual Edit (PUT)
    res_put = client.put(
        f"/api/admin/exams/{exam_id}",
        json={"time_limit": 135, "scoring_rules": "Updated scoring system"},
        headers=get_auth_headers(is_admin=True)
    )
    assert res_put.status_code == 200
    assert res_put.json()["time_limit"] == 135
    assert res_put.json()["scoring_rules"] == "Updated scoring system"

    # 5. Delete Exam
    res_del = client.delete(f"/api/admin/exams/{exam_id}", headers=get_auth_headers(is_admin=True))
    assert res_del.status_code == 200
    assert "deleted successfully" in res_del.json()["message"]

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
