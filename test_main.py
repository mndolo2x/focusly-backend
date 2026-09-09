import os
import io
import jwt
import pytest
from fastapi.testclient import TestClient
from main import app
from services.ollama_service import ollama_service
from services.review_service import review_service
from services.video_service import video_service

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

def test_video_generation_endpoints_flow(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    fake_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"
    pdf_file = io.BytesIO(fake_pdf)

    # 1. Upload Document
    up_res = client.post(
        "/api/documents/upload",
        files={"file": ("history.pdf", pdf_file, "application/pdf")},
        headers=get_auth_headers()
    )
    assert up_res.status_code == 200
    doc_id = up_res.json()["id"]

    # 2. Trigger Video Generation Task
    gen_vid_res = client.post(f"/api/documents/{doc_id}/generate-video", headers=get_auth_headers())
    assert gen_vid_res.status_code == 200
    assert "task_id" in gen_vid_res.json()
    assert gen_vid_res.json()["status"] == "processing"

    # 3. Get Video Status
    status_res = client.get(f"/api/documents/{doc_id}/video-status", headers=get_auth_headers())
    assert status_res.status_code == 200
    assert "status" in status_res.json()

    # 4. Mock Video Lesson Assembly & Fetch Video URL
    import asyncio
    video_url = asyncio.run(video_service.generate_video_lesson(doc_id, title="History Lesson"))
    assert video_url is not None

    get_vid_res = client.get(f"/api/documents/{doc_id}/video", headers=get_auth_headers())
    assert get_vid_res.status_code == 200
    assert "video_url" in get_vid_res.json()

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
