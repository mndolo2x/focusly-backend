import os
import shutil
from typing import Any, Dict, List
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from admin.routes import router as admin_router
from auth import get_current_user, router as auth_router
from config import settings
from models import (
    DocumentResponse,
    SummaryCreate,
    SummaryResponse,
    QuizAttemptCreate,
    QuizAttemptResponse,
    StudyPlanCreate,
    StudyPlanResponse,
    SharedLinkCreate,
    SharedLinkResponse,
    AIContentDisclaimer,
)
from services.document_service import document_service
from services.summary_service import summary_service
from services.quiz_service import quiz_service
from services.video_service import video_service
from services.study_service import study_service
from services.ollama_service import ollama_service
from utils.rag_engine import rag_engine
from workers.tasks import process_document_task

app = FastAPI(
    title="Focusly Backend API",
    description="Local AI Learning Platform converting documents into summaries, quizzes, and narrated videos.",
    version="1.0.0",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(auth_router)
app.include_router(admin_router)

@app.get("/")
async def root():
    return {
        "message": "Focusly Backend API is running",
        "disclaimer": AIContentDisclaimer().disclaimer
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": settings.ENVIRONMENT}

@app.get("/api/health/ollama")
async def ollama_health_check():
    """Health check endpoint for local Ollama LLM server status."""
    return await ollama_service.check_health()

# --- Document Endpoints ---

@app.post("/documents/upload", response_model=Dict[str, Any])
async def upload_document(
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    user_id = user["user_id"]
    upload_dir = "/tmp/focusly_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc_record = await document_service.create_document_record(user_id, file.filename, file_path)

    # Trigger async processing task (Celery task or synchronous fallback)
    try:
        process_document_task.delay(doc_record["id"], file_path)
    except Exception:
        # Fallback inline processing if Celery broker is offline
        from utils.text_extractor import text_extractor
        extracted_text = text_extractor.extract_text_from_file(file_path)
        page_count = text_extractor.get_page_count(file_path)
        doc_record = await document_service.update_document(
            doc_record["id"],
            {"extracted_text": extracted_text, "page_count": page_count, "status": "completed"}
        )
        rag_engine.index_document(doc_record["id"], extracted_text)

    return doc_record

@app.get("/documents", response_model=List[Dict[str, Any]])
async def list_documents(user: Dict[str, Any] = Depends(get_current_user)):
    return await document_service.list_documents(user["user_id"])

@app.get("/documents/{document_id}", response_model=Dict[str, Any])
async def get_document(document_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc

# --- Summary Endpoints ---

@app.post("/summaries", response_model=Dict[str, Any])
async def create_summary(
    payload: SummaryCreate,
    user: Dict[str, Any] = Depends(get_current_user)
):
    quota_ok = await summary_service.check_user_summary_quota(user["user_id"])
    if not quota_ok:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Monthly summary quota exceeded (50/month)")

    doc = await document_service.get_document(payload.document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    text = doc.get("extracted_text", "")
    summary = await summary_service.generate_summary(payload.document_id, text, payload.depth)
    return summary

# --- Quiz Endpoints ---

@app.post("/quizzes/generate/{document_id}", response_model=List[Dict[str, Any]])
async def generate_quiz(
    document_id: str,
    count: int = 5,
    user: Dict[str, Any] = Depends(get_current_user)
):
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    questions = await quiz_service.generate_quiz_questions(document_id, doc.get("extracted_text", ""), count)
    return questions

@app.post("/quizzes/attempt", response_model=Dict[str, Any])
async def submit_quiz_attempt(
    attempt: QuizAttemptCreate,
    user: Dict[str, Any] = Depends(get_current_user)
):
    res = await quiz_service.submit_attempt(
        user["user_id"], attempt.question_id, attempt.selected_answer, attempt.confidence_score
    )
    return res

# --- Video Lesson Endpoints ---

@app.post("/videos/generate/{document_id}", response_model=Dict[str, Any])
async def generate_video(
    document_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    quota_ok = await video_service.check_user_video_quota(user["user_id"])
    if not quota_ok:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Monthly video quota exceeded (10/month)")

    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    sections = [
        {"title": doc.get("filename", "Lesson"), "key_points": ["Key concept overview"], "explanation": doc.get("extracted_text", "")[:300]}
    ]
    video_path = await video_service.generate_video_lesson(document_id, doc.get("filename", "Lesson"), sections)
    return {"document_id": document_id, "status": "video_ready", "video_path": video_path}

# --- Study Plan Endpoints ---

@app.post("/study/plan", response_model=Dict[str, Any])
async def create_study_plan(
    payload: StudyPlanCreate,
    user: Dict[str, Any] = Depends(get_current_user)
):
    plan = await study_service.generate_study_plan(user["user_id"], payload.exam_date, payload.documents or [])
    return plan
