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

def test_timed_exam_mode_flow(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    fake_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"
    pdf_file = io.BytesIO(fake_pdf)

    # 1. Upload Document
    up_res = client.post(
        "/api/documents/upload",
        files={"file": ("physics.pdf", pdf_file, "application/pdf")},
        headers=get_auth_headers()
    )
    assert up_res.status_code == 200
    doc_id = up_res.json()["id"]

    # 2. Start Timed Exam (35 questions)
    start_res = client.post(
        "/api/exam-mode/start",
        json={"subject": "Physics Mechanics", "document_ids": [doc_id], "num_questions": 35},
        headers=get_auth_headers()
    )
    assert start_res.status_code == 200
    exam_session = start_res.json()
    assert "id" in exam_session
    assert exam_session["total_questions"] == 35
    assert exam_session["duration_seconds"] == 4200 # 35 * 120s (~2 mins/question)
    exam_id = exam_session["id"]

    # 3. Check Status
    status_res = client.get("/api/exam-mode/status", headers=get_auth_headers())
    assert status_res.status_code == 200
    s_data = status_res.json()
    assert s_data["time_remaining_seconds"] > 0
    assert s_data["paused"] is False

    # 4. Pause Timer
    pause_res = client.post("/api/exam-mode/pause", json={"exam_id": exam_id}, headers=get_auth_headers())
    assert pause_res.status_code == 200
    assert pause_res.json()["paused"] is True

    # Check status during pause
    status_paused = client.get("/api/exam-mode/status", headers=get_auth_headers())
    assert status_paused.json()["paused"] is True

    # 5. Resume Timer
    resume_res = client.post("/api/exam-mode/resume", json={"exam_id": exam_id}, headers=get_auth_headers())
    assert resume_res.status_code == 200
    assert resume_res.json()["paused"] is False

    # 6. Submit Timed Exam & Grade
    q_id = exam_session["questions"][0]["id"]
    submit_payload = {
        "exam_id": exam_id,
        "answers": [
            {"question_id": q_id, "selected_answer": 0, "confidence_score": 5}
        ]
    }
    submit_res = client.post("/api/exam-mode/submit", json=submit_payload, headers=get_auth_headers())
    assert submit_res.status_code == 200
    grade_data = submit_res.json()
    assert "score" in grade_data
    assert "feedback" in grade_data
    assert grade_data["total_questions"] == 35

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
