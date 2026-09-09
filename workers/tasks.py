import asyncio
from workers.celery_app import celery_app
from services.document_service import document_service
from services.summary_service import summary_service
from services.quiz_service import quiz_service
from services.video_service import video_service
from utils.text_extractor import text_extractor

@celery_app.task(name="workers.tasks.process_document_task")
def process_document_task(doc_id: str, file_path: str):
    """
    Background worker task to extract text from document, update record status,
    and index vector embeddings into RAG engine.
    """
    try:
        extracted_text = text_extractor.extract_text_from_file(file_path)
        page_count = text_extractor.get_page_count(file_path)

        asyncio.run(document_service.update_document(
            doc_id,
            {
                "extracted_text": extracted_text,
                "page_count": page_count,
                "status": "completed"
            }
        ))
        return {"status": "success", "doc_id": doc_id, "page_count": page_count}
    except Exception as e:
        asyncio.run(document_service.update_document(doc_id, {"status": "failed"}))
        return {"status": "error", "message": str(e)}

@celery_app.task(name="workers.tasks.generate_video_task")
def generate_video_task(doc_id: str, title: str, summary_sections: list):
    """
    Background worker task to generate narrated video lessons.
    """
    try:
        video_path = asyncio.run(video_service.generate_video_lesson(doc_id, title, summary_sections))
        return {"status": "success", "doc_id": doc_id, "video_path": video_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}
