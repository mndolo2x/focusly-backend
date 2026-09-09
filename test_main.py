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

def test_user_list_exam_profiles():
    response = client.get("/api/exams", headers=get_auth_headers())
    assert response.status_code == 200
    profiles = response.json()
    assert isinstance(profiles, list)
    assert len(profiles) >= 1

def test_exam_paper_generation_and_grading_flow(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    fake_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"
    pdf_file = io.BytesIO(fake_pdf)

    # 1. Upload Document
    up_res = client.post(
        "/api/documents/upload",
        files={"file": ("sat_prep.pdf", pdf_file, "application/pdf")},
        headers=get_auth_headers()
    )
    assert up_res.status_code == 200
    doc_id = up_res.json()["id"]

    # 2. Generate Practice Exam Paper
    gen_res = client.post(
        "/api/exams/sat-reading/generate-paper",
        json={"document_ids": [doc_id], "num_questions": 10},
        headers=get_auth_headers()
    )
    assert gen_res.status_code == 200
    paper = gen_res.json()
    assert "questions" in paper
    assert len(paper["questions"]) == 10
    assert "This is an unofficial practice paper and is not affiliated with or endorsed by the exam board." in paper["disclaimer_text"]

    paper_id = paper["id"]

    # 3. GET Document Exam Paper
    get_res = client.get(f"/api/documents/{doc_id}/exam-paper", headers=get_auth_headers())
    assert get_res.status_code == 200
    assert "questions" in get_res.json()

    # 4. Submit & Grade Exam Paper
    submissions = {
        "paper_id": paper_id,
        "answers": [
            {"question_id": paper["questions"][0]["id"], "selected_answer": 0}
        ]
    }
    submit_res = client.post("/api/exams/sat-reading/submit-paper", json=submissions, headers=get_auth_headers())
    assert submit_res.status_code == 200
    grade_data = submit_res.json()
    assert "score" in grade_data
    assert "correct_count" in grade_data
    assert "This is an unofficial practice paper and is not affiliated with or endorsed by the exam board." in grade_data["disclaimer_text"]

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
