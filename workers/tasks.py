import asyncio
from workers.celery_app import celery_app
from services.document_service import document_service
from services.summary_service import summary_service
from services.quiz_service import quiz_service
from services.video_service import video_service
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
        # If doc not found via default query, try direct lookup
        if not doc:
            from database import supabase
            res = supabase.table("documents").select("*").eq("id", document_id).execute()
            if res.data:
                doc = res.data[0]

        if not doc:
            return {"status": "error", "message": f"Document {document_id} not found."}

        text = doc.get("extracted_text", "")

        # 2. Chunk text & 3/4. Generate embeddings and store in FAISS index
        asyncio.run(rag_engine.index_document_chunks(document_id, text))

        # 5. Query Ollama for structured summary and 6. Store in summaries table
        summary_record = asyncio.run(summary_service.generate_summary_for_text(document_id, text, depth=depth))

        # 7. Update document status
        asyncio.run(document_service.update_document(document_id, {"status": "completed"}))

        return {"status": "success", "document_id": document_id, "summary_id": summary_record.get("id")}
    except Exception as e:
        asyncio.run(document_service.update_document(document_id, {"status": "failed"}))
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
