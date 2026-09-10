import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from workers.celery_app import celery_app
from services.document_service import document_service
from services.summary_service import summary_service
from services.quiz_service import quiz_service
from services.video_service import video_service
from services.review_service import review_service
from utils.text_extractor import text_extractor
from utils.rag_engine import rag_engine

logger = logging.getLogger("focusly.tasks")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
    logger.addHandler(ch)

@celery_app.task(
    name="workers.tasks.process_image_notes_task",
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
    retry_kwargs={"max_retries": 3, "countdown": 5}
)
def process_image_notes_task(doc_id: str, image_paths: List[str]):
    """
    Background worker task with automatic retries and exponential backoff to process handwritten image notes.
    """
    logger.info(f"Task Started -> process_image_notes_task for doc_id: {doc_id}, images: {len(image_paths)}")
    try:
        res = asyncio.run(document_service.process_image_notes_batch(doc_id, image_paths))
        logger.info(f"Task Success -> process_image_notes_task for doc_id: {doc_id}")
        return {
            "status": "success",
            "doc_id": doc_id,
            "page_count": len(image_paths),
            "ocr_confidence": res.get("ocr_confidence", 0.0),
            "doc_status": res.get("status")
        }
    except Exception as e:
        logger.error(f"Task Error -> process_image_notes_task failed for doc_id {doc_id}: {str(e)}")
        asyncio.run(document_service.update_document(doc_id, {"status": "failed"}))
        raise e

@celery_app.task(
    name="workers.tasks.process_document_task",
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
    retry_kwargs={"max_retries": 3, "countdown": 5}
)
def process_document_task(doc_id: str, file_path: str):
    """
    Background worker task with automatic retries to extract text from document and index into RAG engine.
    """
    logger.info(f"Task Started -> process_document_task for doc_id: {doc_id}")
    try:
        extracted_text, page_count = text_extractor.extract_text_and_page_count(file_path)

        asyncio.run(document_service.update_document(
            doc_id,
            {
                "extracted_text": extracted_text,
                "page_count": page_count,
                "status": "completed"
            }
        ))
        asyncio.run(rag_engine.index_document_chunks(doc_id, extracted_text))
        logger.info(f"Task Success -> process_document_task for doc_id: {doc_id}, pages: {page_count}")
        return {"status": "success", "doc_id": doc_id, "page_count": page_count}
    except Exception as e:
        logger.error(f"Task Error -> process_document_task failed for doc_id {doc_id}: {str(e)}")
        asyncio.run(document_service.update_document(doc_id, {"status": "failed"}))
        raise e

@celery_app.task(
    name="workers.tasks.summarize_document_task",
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
    retry_kwargs={"max_retries": 3, "countdown": 5}
)
def summarize_document_task(document_id: str, depth: str = "standard"):
    """
    Celery task with automatic retries to chunk text, index FAISS vectors, and generate structured summary.
    """
    logger.info(f"Task Started -> summarize_document_task for document_id: {document_id}, depth: {depth}")
    try:
        doc = asyncio.run(document_service.get_document(document_id, user_id=""))
        if not doc:
            from database import supabase
            res = supabase.table("documents").select("*").eq("id", document_id).execute()
            if res.data:
                doc = res.data[0]

        if not doc:
            logger.error(f"Task Error -> summarize_document_task document {document_id} not found.")
            return {"status": "error", "message": f"Document {document_id} not found."}

        text = doc.get("extracted_text", "")
        asyncio.run(rag_engine.index_document_chunks(document_id, text))
        summary_record = asyncio.run(summary_service.generate_summary_for_text(document_id, text, depth=depth))
        asyncio.run(document_service.update_document(document_id, {"status": "completed"}))

        logger.info(f"Task Success -> summarize_document_task for document_id: {document_id}")
        return {"status": "success", "document_id": document_id, "summary_id": summary_record.get("id")}
    except Exception as e:
        logger.error(f"Task Error -> summarize_document_task failed for document_id {document_id}: {str(e)}")
        asyncio.run(document_service.update_document(document_id, {"status": "failed"}))
        raise e

@celery_app.task(
    name="workers.tasks.generate_quiz_task",
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
    retry_kwargs={"max_retries": 3, "countdown": 5}
)
def generate_quiz_task(document_id: str, num_questions: int = 10):
    """
    Celery task with automatic retries to generate quiz questions using Ollama.
    """
    logger.info(f"Task Started -> generate_quiz_task for document_id: {document_id}, questions: {num_questions}")
    try:
        questions = asyncio.run(quiz_service.generate_quiz_from_summary(document_id, num_questions=num_questions))
        logger.info(f"Task Success -> generate_quiz_task generated {len(questions)} questions")
        return {"status": "success", "document_id": document_id, "questions_generated": len(questions)}
    except Exception as e:
        logger.error(f"Task Error -> generate_quiz_task failed for document_id {document_id}: {str(e)}")
        raise e

@celery_app.task(
    name="workers.tasks.generate_video_task",
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
    retry_kwargs={"max_retries": 3, "countdown": 5}
)
def generate_video_task(document_id: str):
    """
    Celery task with 3 retries and exponential backoff to generate narrated MP4 video lesson.
    """
    logger.info(f"Task Started -> generate_video_task for document_id: {document_id}")
    try:
        asyncio.run(document_service.update_document(document_id, {"status": "processing"}))
        video_url = asyncio.run(video_service.generate_video_lesson(document_id))
        logger.info(f"Task Success -> generate_video_task for document_id: {document_id}")
        return {"status": "success", "document_id": document_id, "video_url": video_url}
    except Exception as e:
        logger.error(f"Task Error -> generate_video_task failed for document_id {document_id}: {str(e)}")
        asyncio.run(document_service.update_document(document_id, {"status": "failed"}))
        raise e

@celery_app.task(name="workers.tasks.reset_monthly_usage_task")
def reset_monthly_usage_task():
    """
    Monthly Celery Beat cron task executed on the 1st of every month at 00:00 UTC.
    Resets video_generations_used, summary_generations_used, and quiz_generations_used to 0 for all users.
    """
    now = datetime.utcnow()
    try:
        from database import supabase
        res = supabase.table("user_usage").update({
            "video_generations_used": 0,
            "summary_generations_used": 0,
            "quiz_generations_used": 0,
            "month": now.month,
            "year": now.year
        }).neq("id", "00000000-0000-0000-0000-000000000000").execute()
        reset_count = len(res.data) if res.data else 0
        return {"status": "success", "users_reset": reset_count, "month": now.month, "year": now.year}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@celery_app.task(name="workers.tasks.daily_review_queue_task")
def daily_review_queue_task():
    """
    Daily Celery Beat cron task that finds all review_items where next_review_date <= today
    and queues them for users.
    """
    now_iso = datetime.utcnow().isoformat()
    queued_count = 0

    try:
        from database import supabase
        res = supabase.table("review_items").select("*").lte("next_review_date", now_iso).execute()
        if res.data:
            queued_count = len(res.data)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    return {"status": "success", "items_queued": queued_count, "processed_at": now_iso}
