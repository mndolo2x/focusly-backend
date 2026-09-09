import os
import io
import jwt
import pytest
from fastapi.testclient import TestClient
from main import app
from services.ollama_service import ollama_service

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

def test_auth_me_with_jwt_token(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")
    response = client.get("/api/auth/me", headers=get_auth_headers())
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "user_123"

def test_document_upload_and_summarize_flow(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    fake_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"
    pdf_file = io.BytesIO(fake_pdf)

    # 1. Upload
    up_res = client.post(
        "/api/documents/upload",
        files={"file": ("physics.pdf", pdf_file, "application/pdf")},
        headers=get_auth_headers()
    )
    assert up_res.status_code == 200
    doc_id = up_res.json()["id"]

    # 2. Trigger Summarize Task
    task_res = client.post(f"/api/documents/{doc_id}/summarize?depth=standard", headers=get_auth_headers())
    assert task_res.status_code == 200
    assert "task_id" in task_res.json()

    # 3. Direct Summary
    summary_quick = client.post(f"/api/documents/{doc_id}/summary?depth=quick", headers=get_auth_headers())
    assert summary_quick.status_code == 200

    # 4. Quiz Generation & Retrieval Flow
    gen_quiz_res = client.post(f"/api/documents/{doc_id}/generate-quiz?num_questions=5", headers=get_auth_headers())
    assert gen_quiz_res.status_code == 200
    assert "task_id" in gen_quiz_res.json()

    get_quiz_res = client.get(f"/api/documents/{doc_id}/quiz", headers=get_auth_headers())
    assert get_quiz_res.status_code == 200
    questions = get_quiz_res.json()
    assert isinstance(questions, list)
    assert len(questions) >= 1

    q_id = questions[0]["id"]

    # 5. Quiz Submission
    submissions = [
        {"question_id": q_id, "selected_answer": 0, "confidence_score": 5}
    ]
    submit_res = client.post(f"/api/documents/{doc_id}/quiz/submit", json=submissions, headers=get_auth_headers())
    assert submit_res.status_code == 200
    sub_data = submit_res.json()
    assert "score" in sub_data
    assert "correct_count" in sub_data
    assert "weak_areas" in sub_data

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
