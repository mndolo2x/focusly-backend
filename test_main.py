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

def test_public_share_links_flow(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    # 1. Upload Document
    fake_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"
    up_res = client.post(
        "/api/documents/upload",
        files={"file": ("biology_notes.pdf", io.BytesIO(fake_pdf), "application/pdf")},
        headers=get_auth_headers()
    )
    assert up_res.status_code == 200
    doc_id = up_res.json()["id"]

    # Mock summary and quiz questions
    async def mock_summary(document_id):
        return {
            "id": "sum_1",
            "document_id": document_id,
            "sections": [
                {"title": "Cell Biology", "content": "Cells are the basic unit of life.", "bullets": ["Prokaryotes", "Eukaryotes"]}
            ]
        }

    async def mock_questions(document_id):
        return [
            {
                "id": "q1",
                "document_id": document_id,
                "section_index": 0,
                "question_text": "What is the powerhouse of the cell?",
                "options": ["Mitochondria", "Nucleus", "Ribosome", "Golgi"],
                "correct_answer_index": 0
            }
        ]

    monkeypatch.setattr("services.summary_service.summary_service.get_latest_summary", mock_summary)
    monkeypatch.setattr("services.quiz_service.quiz_service.get_quiz_questions", mock_questions)

    # 2. Create Share Link (Authenticated)
    share_res = client.post(
        f"/api/documents/{doc_id}/share",
        json={"expires_in_days": 7},
        headers=get_auth_headers()
    )
    assert share_res.status_code == 200
    share_data = share_res.json()
    assert "share_id" in share_data
    share_id = share_data["share_id"]

    # 3. Get Public Shared Content (Unauthenticated)
    pub_res = client.get(f"/api/public/{share_id}")
    assert pub_res.status_code == 200
    pub_data = pub_res.json()

    assert pub_data["title"] == "biology_notes.pdf"
    assert len(pub_data["summary_sections"]) == 1
    assert pub_data["summary_sections"][0]["title"] == "Cell Biology"
    assert len(pub_data["quiz_questions"]) == 1
    q1 = pub_data["quiz_questions"][0]
    assert q1["question_text"] == "What is the powerhouse of the cell?"
    # EXCLUSION CHECKS: No video_url, no full extracted_text, no user_id, no correct_answer_index in public questions
    assert "video_url" not in pub_data
    assert "extracted_text" not in pub_data
    assert "user_id" not in pub_data
    assert "correct_answer_index" not in q1
    assert "disclaimer" in pub_data

    # 4. Anonymous Quiz Submit (Unauthenticated)
    sub_res = client.post(
        f"/api/public/{share_id}/quiz/submit",
        json={
            "submissions": [
                {"question_id": "q1", "selected_answer": 0}
            ]
        }
    )
    assert sub_res.status_code == 200
    sub_data = sub_res.json()
    assert sub_data["score_percent"] == 100.0
    assert sub_data["correct_count"] == 1
    assert sub_data["total_questions"] == 1
    assert "disclaimer" in sub_data

def test_vision_health_check(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    async def mock_ollama_health():
        return {"status": "healthy", "models": ["llama3.2-vision:11b", "nomic-embed-text"]}

    monkeypatch.setattr("services.ollama_service.ollama_service.check_health", mock_ollama_health)

    res = client.get("/api/health/vision")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["vision_available"] is True
    assert "llama3.2-vision" in data["vision_model"]

def test_image_notes_upload_and_status_flow(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    from PIL import Image, ImageDraw
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Handwritten note content", fill=(0, 0, 0))

    img_bytes_io = io.BytesIO()
    img.save(img_bytes_io, format="JPEG")
    img_bytes = img_bytes_io.getvalue()

    # Mock OCR response
    async def mock_transcribe(image_input, page_num=1):
        return {
            "virtual_page": page_num,
            "text": f"--- Virtual Page {page_num} ---\nPhysics formulas\n[DIAGRAM: Circuit diagram with resistor and battery]",
            "raw_text": "Physics formulas\n[DIAGRAM: Circuit diagram with resistor and battery]",
            "confidence": 0.88,
            "ocr_tier": "llama3.2-vision",
            "diagrams": ["Circuit diagram with resistor and battery"],
            "is_blurry": False,
            "is_low_confidence": False,
            "blur_score": 150.0
        }

    monkeypatch.setattr("utils.handwriting_ocr.handwriting_ocr.transcribe_handwritten_image", mock_transcribe)

    upload_res = client.post(
        "/api/documents/upload-image",
        files=[
            ("files", ("page1.jpg", io.BytesIO(img_bytes), "image/jpeg")),
            ("files", ("page2.jpg", io.BytesIO(img_bytes), "image/jpeg"))
        ],
        headers=get_auth_headers()
    )

    assert upload_res.status_code == 200
    res_data = upload_res.json()
    assert "id" in res_data
    doc_id = res_data["id"]
    assert res_data["source_type"] == "image_notes"
    assert res_data["page_count"] == 2
    assert res_data["ocr_confidence"] == 0.88
    assert len(res_data["virtual_page_map"]) == 2
    assert res_data["virtual_page_map"][0]["virtual_page"] == 1
    assert "Circuit diagram" in res_data["virtual_page_map"][0]["diagrams_extracted"][0]

    # Check status endpoint
    status_res = client.get(f"/api/documents/{doc_id}/status", headers=get_auth_headers())
    assert status_res.status_code == 200
    st_data = status_res.json()
    assert st_data["id"] == doc_id
    assert st_data["source_type"] == "image_notes"
    assert st_data["page_count"] == 2
    assert len(st_data["virtual_page_map"]) == 2

def test_image_notes_low_confidence_flagging(monkeypatch):
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", "test_secret")

    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(100, 100, 100))
    img_bytes_io = io.BytesIO()
    img.save(img_bytes_io, format="JPEG")
    img_bytes = img_bytes_io.getvalue()

    # Mock low confidence OCR result
    async def mock_low_conf_transcribe(image_input, page_num=1):
        return {
            "virtual_page": page_num,
            "text": f"--- Virtual Page {page_num} ---\nunreadable scribble",
            "raw_text": "unreadable scribble",
            "confidence": 0.35,
            "ocr_tier": "tesseract",
            "diagrams": [],
            "is_blurry": True,
            "is_low_confidence": True,
            "blur_score": 40.0
        }

    monkeypatch.setattr("utils.handwriting_ocr.handwriting_ocr.transcribe_handwritten_image", mock_low_conf_transcribe)

    upload_res = client.post(
        "/api/documents/upload-image",
        files=[("files", ("blurry.jpg", io.BytesIO(img_bytes), "image/jpeg"))],
        headers=get_auth_headers()
    )

    assert upload_res.status_code == 200
    res_data = upload_res.json()
    assert res_data["status"] == "low_confidence"
    assert res_data["ocr_confidence"] == 0.35
    assert "low OCR confidence" in res_data["message"]

@pytest.mark.anyio
async def test_ollama_service_generate_retry_error_handling(monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        await ollama_service.generate("Test prompt", max_retries=2, retry_delay=0.01)
    assert "Ollama connection error" in str(exc_info.value) or "Ollama" in str(exc_info.value)
