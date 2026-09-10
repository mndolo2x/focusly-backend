import os
import shutil
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, Query, Response, status
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
    ExamPaperGenerateRequest,
    ExamPaperSubmission,
    ExamPaperResponse,
    ExamPaperGradeResponse,
    EXAM_PAPER_DISCLAIMER,
    StudyPlanCreateAI,
    StudyPlanUpdateAI,
    DocumentMergeRequest,
)
from services.document_service import document_service
from services.summary_service import summary_service
from services.quiz_service import quiz_service
from services.review_service import review_service
from services.video_service import video_service
from services.study_service import study_service
from services.ollama_service import ollama_service
from services.tts_service import tts_service
from utils.text_extractor import text_extractor
from utils.rag_engine import rag_engine
from workers.tasks import process_document_task, summarize_document_task, generate_quiz_task, generate_video_task, daily_review_queue_task

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

@app.get("/api/health/tts")
async def tts_health_check():
    """Health check endpoint for local Kokoro TTS engine status."""
    return await tts_service.check_health()

# --- Dashboard Progress Endpoint ---

@app.get("/api/dashboard/progress")
async def get_dashboard_progress(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Returns user learning progress metrics (cached for 5 minutes):
    total_lessons, completed_lessons, average_quiz_score, quiz_score_history,
    weak_areas, reviews_due_today, upcoming_reviews, study_plan, exam_countdown.
    """
    return await study_service.get_dashboard_progress(user["user_id"])

# --- AI Study Plan Builder Endpoints ---

@app.post("/api/study-plan")
async def create_study_plan_endpoint(
    payload: StudyPlanCreateAI,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Creates an AI-powered study plan calculating days remaining, prioritizing weak areas,
    and generating a day-by-day task schedule.
    """
    return await study_service.create_ai_study_plan(
        user["user_id"], payload.exam_date, payload.subject, payload.documents or []
    )

@app.get("/api/study-plan")
async def get_study_plan_endpoint(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns current active study plan for user."""
    plan = await study_service.get_user_study_plan(user["user_id"])
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active study plan found")
    return plan

@app.put("/api/study-plan")
async def update_study_plan_endpoint(
    payload: StudyPlanUpdateAI,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Regenerates/updates current study plan."""
    return await study_service.update_ai_study_plan(
        user["user_id"], payload.exam_date, payload.subject, payload.documents
    )

@app.get("/api/study-plan/today")
async def get_today_study_tasks_endpoint(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns today's scheduled study tasks."""
    return await study_service.get_today_study_tasks(user["user_id"])

@app.post("/api/study-plan/tasks/{task_id}/complete")
async def complete_study_task_endpoint(
    task_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Marks a specific study task as complete."""
    return await study_service.complete_study_task(user["user_id"], task_id)

# --- Public Exam Profile Endpoints ---

@app.get("/api/exams")
@app.get("/exams")
async def list_user_exam_profiles():
    """Returns available exam profiles for users."""
    from admin.routes import list_all_exam_profiles
    return await list_all_exam_profiles()

@app.post("/api/exams/{exam_id}/generate-paper", response_model=Dict[str, Any])
async def generate_exam_paper_endpoint(
    exam_id: str,
    payload: ExamPaperGenerateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Uses exam profile + Ollama to generate practice questions matching the exact exam format.
    CRITICAL: Response includes disclaimer: "This is an unofficial practice paper and is not affiliated with or endorsed by the exam board."
    """
    paper = await study_service.generate_exam_paper(
        user["user_id"], exam_id, payload.document_ids, payload.num_questions
    )
    return paper

@app.get("/api/documents/{document_id}/exam-paper")
async def get_document_exam_paper_endpoint(
    document_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns generated practice exam paper for a specific document."""
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    paper = await study_service.get_document_exam_paper(document_id)
    if not paper:
        paper = await study_service.generate_exam_paper(user["user_id"], "practice_exam", [document_id], num_questions=25)

    return paper

@app.post("/api/exams/{exam_id}/submit-paper")
async def submit_exam_paper_endpoint(
    exam_id: str,
    payload: Dict[str, Any],
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Grades submitted paper and returns score breakdown with mandatory disclaimer text.
    """
    paper_id = payload.get("paper_id") or exam_id
    answers = payload.get("answers") or payload.get("submissions") or []

    results = await study_service.submit_exam_paper(user["user_id"], paper_id, answers)
    return results

# --- Document Endpoints ---

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

@app.post("/api/documents/upload", response_model=Dict[str, Any])
@app.post("/documents/upload", response_model=Dict[str, Any])
async def upload_document(
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Validates PDF file, uploads to Supabase Storage, extracts text via PyMuPDF/OCR/pdfplumber,
    stores record in database, and sets status='completed'.
    """
    user_id = user["user_id"]

    filename = file.filename or "uploaded.pdf"
    if not (filename.lower().endswith(".pdf") or file.content_type in ["application/pdf", "octet-stream"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF documents are allowed."
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum allowed limit of 50MB ({len(file_bytes)} bytes)."
        )

    file_url = await document_service.upload_file_to_storage(file_bytes, filename, user_id)

    upload_dir = "/tmp/focusly_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    local_file_path = os.path.join(upload_dir, f"{user_id}_{filename}")
    with open(local_file_path, "wb") as buffer:
        buffer.write(file_bytes)

    doc_record = await document_service.create_document_record(user_id, filename, file_url)
    doc_id = doc_record["id"]

    try:
        extracted_text, page_count = text_extractor.extract_text_and_page_count(local_file_path)
        doc_record = await document_service.update_document(
            doc_id,
            {
                "extracted_text": extracted_text,
                "page_count": page_count,
                "status": "completed",
                "file_url": file_url
            }
        )
        await rag_engine.index_document_chunks(doc_id, extracted_text)
    except Exception as e:
        doc_record = await document_service.update_document(doc_id, {"status": "failed"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document text extraction failed: {str(e)}"
        )

    return doc_record

@app.post("/api/documents/merge", response_model=Dict[str, Any])
async def merge_documents_endpoint(
    payload: DocumentMergeRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Merges multiple user documents:
    1. Validates ownership of all documents.
    2. Combines extracted text preserving offset page numbers.
    3. Creates new document entry with status='processing'.
    4. Triggers background summary generation.
    """
    try:
        merged_doc = await document_service.merge_documents(
            user["user_id"], payload.document_ids, payload.merged_filename
        )
        return merged_doc
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Merge failed: {str(e)}")

@app.get("/api/documents", response_model=Dict[str, Any])
@app.get("/documents", response_model=Dict[str, Any])
async def list_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns paginated list of user's documents."""
    return await document_service.list_documents_paginated(user["user_id"], page=page, limit=limit)

@app.get("/api/documents/{document_id}", response_model=Dict[str, Any])
@app.get("/documents/{document_id}", response_model=Dict[str, Any])
async def get_document(document_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Returns specific document record along with its generated summary if present."""
    doc_with_summary = await document_service.get_document_with_summary(document_id, user["user_id"])
    if not doc_with_summary:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc_with_summary

@app.delete("/api/documents/{document_id}")
@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Deletes document record and removes file from Supabase Storage."""
    deleted = await document_service.delete_document(document_id, user["user_id"])
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return {"message": "Document and associated storage file deleted successfully", "id": document_id}

# --- Summary Endpoints ---

@app.post("/api/documents/{document_id}/summarize")
async def trigger_summarize_task(
    document_id: str,
    depth: str = Query("standard", pattern="^(quick|standard|deep)$"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Triggers Celery background task to summarize document with FAISS source tracking."""
    quota_ok = await summary_service.check_user_summary_quota(user["user_id"])
    if not quota_ok:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Monthly summary quota exceeded (50/month)")

    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        task = summarize_document_task.delay(document_id, depth=depth)
        task_id = task.id
    except Exception:
        task_id = str(uuid.uuid4())
        text = doc.get("extracted_text", "")
        await rag_engine.index_document_chunks(document_id, text)
        await summary_service.generate_summary_for_text(document_id, text, depth=depth)

    return {"task_id": task_id, "document_id": document_id, "status": "processing", "depth": depth}

@app.get("/api/documents/{document_id}/summary")
async def get_latest_document_summary(
    document_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns the most recent summary for a specific document."""
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    summary = await summary_service.get_latest_summary(document_id)
    if not summary:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No summary found for this document")

    return summary

@app.post("/api/documents/{document_id}/summary")
@app.post("/summaries", response_model=Dict[str, Any])
async def create_summary(
    document_id: Optional[str] = None,
    payload: Optional[SummaryCreate] = None,
    depth: str = Query("standard", pattern="^(quick|standard|deep)$"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Generates concise or comprehensive summary directly based on depth query parameter."""
    target_doc_id = document_id or (payload.document_id if payload else None)
    if not target_doc_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing document_id")

    quota_ok = await summary_service.check_user_summary_quota(user["user_id"])
    if not quota_ok:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Monthly summary quota exceeded (50/month)")

    doc = await document_service.get_document(target_doc_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if payload and hasattr(payload, 'depth') and payload.depth:
        depth = payload.depth.value if hasattr(payload.depth, 'value') else str(payload.depth)

    text = doc.get("extracted_text", "")
    await rag_engine.index_document_chunks(target_doc_id, text)
    summary = await summary_service.generate_summary_for_text(target_doc_id, text, depth=depth)
    return summary

# --- Quiz Endpoints ---

@app.post("/api/documents/{document_id}/generate-quiz")
async def trigger_generate_quiz(
    document_id: str,
    num_questions: int = Query(10, ge=1, le=30),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Triggers Celery background task to generate multiple-choice quiz from summary."""
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        task = generate_quiz_task.delay(document_id, num_questions=num_questions)
        task_id = task.id
    except Exception:
        task_id = str(uuid.uuid4())
        await quiz_service.generate_quiz_from_summary(document_id, num_questions=num_questions)

    return {"task_id": task_id, "document_id": document_id, "num_questions": num_questions, "status": "processing"}

@app.get("/api/documents/{document_id}/quiz")
async def get_document_quiz(
    document_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns all quiz questions for a document."""
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    questions = await quiz_service.get_quiz_questions(document_id)
    return questions

@app.post("/api/documents/{document_id}/quiz/submit")
async def submit_document_quiz(
    document_id: str,
    submissions: List[Dict[str, Any]],
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Evaluates quiz submission array [{question_id, selected_answer, confidence_score}].
    Returns score percentage, correct/incorrect breakdown, and identified weak areas.
    """
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    results = await quiz_service.submit_quiz_batch(user["user_id"], document_id, submissions)
    return results

# --- Weak Area Tracking Endpoints ---

@app.get("/api/documents/{document_id}/weak-areas")
async def get_document_weak_areas(
    document_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns sections/pages where user answered incorrectly with miss counts and priority scores."""
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    weak_areas = await quiz_service.get_document_weak_areas(user["user_id"], document_id)
    return weak_areas

@app.post("/api/documents/{document_id}/weak-areas/review")
async def mark_weak_area_reviewed(
    document_id: str,
    payload: Dict[str, Any],
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Marks a specific weak area (section_index and page_reference) as reviewed."""
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    sec_idx = payload.get("section_index", 0)
    page_ref = payload.get("page_reference", 1)

    await quiz_service.mark_weak_area_reviewed(user["user_id"], document_id, sec_idx, page_ref)
    return {
        "message": "Weak area marked as reviewed successfully",
        "document_id": document_id,
        "section_index": sec_idx,
        "page_reference": page_ref
    }

@app.get("/api/dashboard/weak-areas")
async def get_dashboard_weak_areas(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Aggregates weak areas across ALL user documents, sorted by most-missed/priority score."""
    return await quiz_service.get_dashboard_weak_areas(user["user_id"])

# --- Spaced Repetition Review Endpoints ---

@app.get("/api/reviews/today")
async def get_reviews_today(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns questions due for spaced repetition review today."""
    due_items = await review_service.get_due_reviews_today(user["user_id"])
    return {"user_id": user["user_id"], "due_count": len(due_items), "items": due_items}

@app.post("/api/reviews/submit")
async def submit_review_item(
    payload: Dict[str, Any],
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Accepts {review_item_id, quality (0-5)} and updates SM-2 parameters and next review date."""
    review_item_id = payload.get("review_item_id")
    quality = payload.get("quality", 3)

    if not review_item_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing review_item_id")

    if not isinstance(quality, int) or not (0 <= quality <= 5):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Quality score must be an integer between 0 and 5")

    updated_item = await review_service.submit_review_item(user["user_id"], review_item_id, quality)
    return updated_item

@app.get("/api/reviews/upcoming")
async def get_upcoming_reviews(
    days: int = Query(7, ge=1, le=30),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns upcoming reviews for the next N days (default 7 days)."""
    return await review_service.get_upcoming_reviews(user["user_id"], days=days)

# --- Video Lesson Endpoints ---

@app.post("/api/documents/{document_id}/generate-video")
@app.post("/videos/generate/{document_id}")
async def generate_video(
    document_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Triggers video lesson generation task."""
    quota_ok = await video_service.check_user_video_quota(user["user_id"])
    if not quota_ok:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Monthly video quota exceeded (10/month)")

    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        task = generate_video_task.delay(document_id)
        task_id = task.id
    except Exception:
        task_id = str(uuid.uuid4())
        await video_service.generate_video_lesson(document_id)

    return {"task_id": task_id, "document_id": document_id, "status": "processing"}

@app.get("/api/documents/{document_id}/video-status")
async def get_document_video_status(
    document_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns current video generation status ('processing', 'ready', 'video_ready', or 'failed')."""
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    doc_status = doc.get("status", "processing")
    video_state = "ready" if doc_status == "video_ready" else doc_status
    return {"document_id": document_id, "status": video_state, "raw_status": doc_status}

@app.get("/api/documents/{document_id}/video")
async def get_document_video_url(
    document_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns public video URL if lesson video is ready."""
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    video_url = await video_service.get_video_url(document_id)
    if not video_url:
        if doc.get("status") == "video_ready" and doc.get("file_url"):
            video_url = doc.get("file_url")

    if not video_url:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not ready for this document")

    return {"document_id": document_id, "video_url": video_url, "status": "video_ready"}

@app.get("/api/documents/{document_id}/transcript")
async def get_document_video_transcript(
    document_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns section-by-section transcript with timestamps and page tracking for document video."""
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    transcript = await video_service.get_document_transcript(document_id)
    return {"document_id": document_id, "sections": transcript}

@app.get("/api/documents/{document_id}/captions")
async def get_document_video_captions(
    document_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns WebVTT formatted caption text for document video."""
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    vtt_content = await video_service.generate_webvtt_captions(document_id)
    return Response(content=vtt_content, media_type="text/vtt")

# Legacy Quiz compatibility endpoints
@app.post("/quizzes/generate/{document_id}", response_model=List[Dict[str, Any]])
async def legacy_generate_quiz(
    document_id: str,
    count: int = 5,
    user: Dict[str, Any] = Depends(get_current_user)
):
    doc = await document_service.get_document(document_id, user["user_id"])
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    questions = await quiz_service.generate_quiz_from_summary(document_id, num_questions=count)
    return questions

@app.post("/quizzes/attempt", response_model=Dict[str, Any])
async def legacy_submit_quiz_attempt(
    attempt: QuizAttemptCreate,
    user: Dict[str, Any] = Depends(get_current_user)
):
    res = await quiz_service.submit_attempt(
        user["user_id"], attempt.question_id, attempt.selected_answer, attempt.confidence_score
    )
    return res

# --- Study Plan Endpoints ---

@app.post("/study/plan", response_model=Dict[str, Any])
async def create_study_plan(
    payload: StudyPlanCreate,
    user: Dict[str, Any] = Depends(get_current_user)
):
    plan = await study_service.generate_study_plan(user["user_id"], payload.exam_date, payload.documents or [])
    return plan
