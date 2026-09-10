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

def test_document_merge_flow(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    fake_pdf1 = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"
    pdf_file1 = io.BytesIO(fake_pdf1)

    fake_pdf2 = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"
    pdf_file2 = io.BytesIO(fake_pdf2)

    # 1. Upload Doc 1
    up_res1 = client.post(
        "/api/documents/upload",
        files={"file": ("part1.pdf", pdf_file1, "application/pdf")},
        headers=get_auth_headers()
    )
    assert up_res1.status_code == 200
    doc1_id = up_res1.json()["id"]

    # 2. Upload Doc 2
    up_res2 = client.post(
        "/api/documents/upload",
        files={"file": ("part2.pdf", pdf_file2, "application/pdf")},
        headers=get_auth_headers()
    )
    assert up_res2.status_code == 200
    doc2_id = up_res2.json()["id"]

    # 3. Merge Documents (POST /api/documents/merge)
    merge_res = client.post(
        "/api/documents/merge",
        json={"document_ids": [doc1_id, doc2_id], "merged_filename": "Combined_Biology.pdf"},
        headers=get_auth_headers()
    )
    assert merge_res.status_code == 200
    merged_data = merge_res.json()
    assert "id" in merged_data
    assert merged_data["filename"] == "Combined_Biology.pdf"
    assert merged_data["id"] != doc1_id
    assert merged_data["id"] != doc2_id
    assert merged_data["status"] == "processing" or merged_data["status"] == "completed"

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
