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

def test_video_transcript_and_captions_flow(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    fake_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"
    pdf_file = io.BytesIO(fake_pdf)

    # Upload
    up_res = client.post(
        "/api/documents/upload",
        files={"file": ("civics.pdf", pdf_file, "application/pdf")},
        headers=get_auth_headers()
    )
    assert up_res.status_code == 200
    doc_id = up_res.json()["id"]

    # Generate Video & Transcript
    import asyncio
    asyncio.run(video_service.generate_video_lesson(doc_id, title="Civics Lesson"))

    # GET Transcript
    trans_res = client.get(f"/api/documents/{doc_id}/transcript", headers=get_auth_headers())
    assert trans_res.status_code == 200
    t_data = trans_res.json()
    assert "sections" in t_data
    assert len(t_data["sections"]) >= 1
    assert "start_time" in t_data["sections"][0]
    assert "end_time" in t_data["sections"][0]

    # GET WebVTT Captions
    vtt_res = client.get(f"/api/documents/{doc_id}/captions", headers=get_auth_headers())
    assert vtt_res.status_code == 200
    assert "WEBVTT" in vtt_res.text
    assert "-->" in vtt_res.text

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
