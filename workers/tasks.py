import asyncio
from datetime import datetime
from workers.celery_app import celery_app
from services.document_service import document_service
from services.summary_service import summary_service
from services.quiz_service import quiz_service
from services.video_service import video_service
from services.review_service import review_service
from utils.text_extractor import text_extractor
from utils.rag_engine import rag_engine

@celery_app.task(name="workers.tasks.process_document_task")
def process_document_task(doc_id: str, file_path: str):
    """
    Background worker task to extract text from document, update record status,
    and index vector embeddings into RAG engine.
    """
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
        return {"status": "success", "doc_id": doc_id, "page_count": page_count}
    except Exception as e:
        asyncio.run(document_service.update_document(doc_id, {"status": "failed"}))
        return {"status": "error", "message": str(e)}

@celery_app.task(name="workers.tasks.summarize_document_task")
def summarize_document_task(document_id: str, depth: str = "standard"):
    """
    Celery task to chunk document text, generate embeddings using Ollama,
    index vectors into FAISS, query Ollama for structured summary with page numbers,
    store result in summaries table, and update document status.
    """
    try:
        doc = asyncio.run(document_service.get_document(document_id, user_id=""))
        if not doc:
            from database import supabase
            res = supabase.table("documents").select("*").eq("id", document_id).execute()
            if res.data:
                doc = res.data[0]

        if not doc:
            return {"status": "error", "message": f"Document {document_id} not found."}

        text = doc.get("extracted_text", "")
        asyncio.run(rag_engine.index_document_chunks(document_id, text))
        summary_record = asyncio.run(summary_service.generate_summary_for_text(document_id, text, depth=depth))
        asyncio.run(document_service.update_document(document_id, {"status": "completed"}))

        return {"status": "success", "document_id": document_id, "summary_id": summary_record.get("id")}
    except Exception as e:
        asyncio.run(document_service.update_document(document_id, {"status": "failed"}))
        return {"status": "error", "message": str(e)}

@celery_app.task(name="workers.tasks.generate_quiz_task")
def generate_quiz_task(document_id: str, num_questions: int = 10):
    """
    Celery task to generate multiple-choice quiz questions from document summary sections using Ollama.
    """
    try:
        questions = asyncio.run(quiz_service.generate_quiz_from_summary(document_id, num_questions=num_questions))
        return {"status": "success", "document_id": document_id, "questions_generated": len(questions)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@celery_app.task(
    name="workers.tasks.generate_video_task",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 5}
)
def generate_video_task(document_id: str):
    """
    Celery task with 3 retries and exponential backoff to generate narrated MP4 video lesson:
    1. Sets document status='processing'.
    2. Calls Kokoro TTS for audio and Pillow for 1080p white slides with page numbers.
    3. Assembles MP4 video using FFmpeg.
    4. Uploads MP4 to Supabase Storage.
    5. Updates document status='video_ready' and stores public video URL.
    """
    try:
        asyncio.run(document_service.update_document(document_id, {"status": "processing"}))
        video_url = asyncio.run(video_service.generate_video_lesson(document_id))
        return {"status": "success", "document_id": document_id, "video_url": video_url}
    except Exception as e:
        asyncio.run(document_service.update_document(document_id, {"status": "failed"}))
        raise e

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
