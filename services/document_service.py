import re
import uuid
from typing import Any, Dict, List, Optional
from database import supabase
from models import DocumentStatus

# In-memory document store for local development/testing without live Supabase database
_in_memory_docs: Dict[str, Dict[str, Any]] = {}

class DocumentService:
    """Service for managing document uploads, database state, Supabase storage, and text processing."""

    async def upload_file_to_storage(self, file_bytes: bytes, filename: str, user_id: str) -> str:
        """Uploads file bytes to Supabase Storage 'documents' bucket and returns storage URL or path."""
        storage_path = f"{user_id}/{uuid.uuid4().hex}_{filename}"
        try:
            supabase.storage.from_("documents").upload(
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": "application/pdf"}
            )
            public_url = supabase.storage.from_("documents").get_public_url(storage_path)
            return public_url or storage_path
        except Exception:
            return storage_path

    async def create_document_record(
        self, user_id: str, filename: str, file_url: Optional[str] = None, page_count: int = 0, source_type: str = "pdf"
    ) -> Dict[str, Any]:
        """Creates a document database entry with status 'processing'."""
        doc_id = str(uuid.uuid4())
        record = {
            "id": doc_id,
            "user_id": user_id,
            "filename": filename,
            "file_url": file_url or "",
            "extracted_text": "",
            "status": DocumentStatus.PROCESSING,
            "page_count": page_count,
            "source_type": source_type,
            "original_images": [],
            "virtual_page_map": [],
        }
        _in_memory_docs[doc_id] = record
        try:
            res = supabase.table("documents").insert(record).execute()
            if res.data:
                _in_memory_docs[doc_id] = res.data[0]
                return res.data[0]
        except Exception:
            pass
        return record

    async def update_document(self, doc_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Updates document record in Supabase or in-memory store."""
        if doc_id in _in_memory_docs:
            _in_memory_docs[doc_id].update(updates)

        try:
            res = supabase.table("documents").update(updates).eq("id", doc_id).execute()
            if res.data:
                _in_memory_docs[doc_id] = res.data[0]
                return res.data[0]
        except Exception:
            pass
        return _in_memory_docs.get(doc_id, {"id": doc_id, **updates})

    async def get_document(self, doc_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves user's document record."""
        try:
            res = supabase.table("documents").select("*").eq("id", doc_id).eq("user_id", user_id).execute()
            if res.data:
                return res.data[0]
        except Exception:
            pass

        doc = _in_memory_docs.get(doc_id)
        if doc and (doc.get("user_id") == user_id or not user_id):
            return doc
        return None

    async def get_document_with_summary(self, doc_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves user's document along with its generated summary."""
        doc = await self.get_document(doc_id, user_id)
        if not doc:
            return None

        summary = None
        try:
            res = supabase.table("summaries").select("*").eq("document_id", doc_id).execute()
            if res.data:
                summary = res.data[0]
        except Exception:
            pass

        return {**doc, "summary": summary}

    async def process_image_notes_batch(
        self, doc_id: str, image_paths: List[str]
    ) -> Dict[str, Any]:
        """
        Processes a batch of uploaded image notes:
        1. Preprocesses and transcribes each image with HandwritingOCR.
        2. Assigns virtual page numbers (1..N).
        3. Builds virtual_page_map with metadata, diagrams, and confidence scores.
        4. Calculates aggregate confidence.
        5. Updates document status ('completed' or 'low_confidence').
        6. Triggers summary generation.
        """
        from utils.handwriting_ocr import handwriting_ocr
        from utils.rag_engine import rag_engine
        from services.summary_service import summary_service

        virtual_page_map = []
        combined_text_blocks = []
        total_confidence = 0.0
        low_confidence_pages = []

        for idx, img_path in enumerate(image_paths, start=1):
            res = await handwriting_ocr.transcribe_handwritten_image(img_path, page_num=idx)
            page_text = res["text"]
            conf = res["confidence"]
            diagrams = res["diagrams"]
            is_blurry = res["is_blurry"]

            total_confidence += conf
            if res["is_low_confidence"]:
                low_confidence_pages.append(idx)

            virtual_page_map.append({
                "virtual_page": idx,
                "image_path": img_path,
                "ocr_confidence": conf,
                "ocr_tier": res["ocr_tier"],
                "diagrams_extracted": diagrams,
                "is_blurry": is_blurry,
                "low_confidence_warning": res["is_low_confidence"]
            })

            combined_text_blocks.append(page_text)

        final_extracted_text = "\n\n".join(combined_text_blocks)
        page_count = len(image_paths)
        avg_confidence = round(total_confidence / page_count, 2) if page_count > 0 else 0.0

        final_status = DocumentStatus.LOW_CONFIDENCE if (avg_confidence < 0.60 or len(low_confidence_pages) > 0) else DocumentStatus.COMPLETED

        updates = {
            "extracted_text": final_extracted_text,
            "page_count": page_count,
            "status": final_status,
            "source_type": "image_notes",
            "original_images": image_paths,
            "virtual_page_map": virtual_page_map,
        }

        updated_doc = await self.update_document(doc_id, updates)

        # Index text and trigger summary generation
        try:
            await rag_engine.index_document_chunks(doc_id, final_extracted_text)
            from workers.tasks import summarize_document_task
            summarize_document_task.delay(doc_id, depth="standard")
        except Exception:
            await summary_service.generate_summary_for_text(doc_id, final_extracted_text, depth="standard")

        return {
            **updated_doc,
            "ocr_confidence": avg_confidence,
            "low_confidence_pages": low_confidence_pages
        }

    async def merge_documents(
        self, user_id: str, document_ids: List[str], merged_filename: Optional[str] = "Merged_Document.pdf"
    ) -> Dict[str, Any]:
        """
        Validates ownership of all requested documents, combines extracted_text,
        preserves and offsets page numbers, creates new document entry with status='processing',
        and triggers summary generation.
        """
        if not document_ids:
            raise ValueError("No document_ids provided for merge.")

        fetched_docs = []
        for doc_id in document_ids:
            doc = await self.get_document(doc_id, user_id)
            if not doc:
                raise ValueError(f"Document {doc_id} not found or does not belong to user.")
            fetched_docs.append(doc)

        merged_filename = merged_filename or "Merged_Document.pdf"
        if not merged_filename.endswith(".pdf"):
            merged_filename += ".pdf"

        # Create new merged document record with status='processing'
        merged_doc_record = await self.create_document_record(user_id, merged_filename, page_count=0)
        merged_id = merged_doc_record["id"]

        combined_text_blocks = []
        cumulative_page_offset = 0

        for idx, doc in enumerate(fetched_docs):
            doc_text = doc.get("extracted_text", "")
            doc_pages = doc.get("page_count", 0) or 1

            # Offset page markers in document text (e.g. '--- Page 1 ---')
            def _offset_page_match(match):
                p_num = int(match.group(1))
                return f"--- Page {cumulative_page_offset + p_num} ---"

            offset_text = re.sub(r'---\s*Page\s*(\d+)\s*---', _offset_page_match, doc_text)

            if not re.search(r'---\s*Page\s*\d+\s*---', offset_text) and offset_text.strip():
                offset_text = f"--- Page {cumulative_page_offset + 1} ---\n" + offset_text

            combined_text_blocks.append(f"=== Document {idx+1}: {doc.get('filename', 'Source')} ===\n{offset_text}")
            cumulative_page_offset += doc_pages

        final_extracted_text = "\n\n".join(combined_text_blocks)

        # Update merged document record with final extracted text and page count
        merged_doc_record = await self.update_document(
            merged_id,
            {
                "extracted_text": final_extracted_text,
                "page_count": cumulative_page_offset,
                "status": DocumentStatus.PROCESSING
            }
        )

        # Trigger summary generation on merged document
        try:
            from workers.tasks import summarize_document_task
            summarize_document_task.delay(merged_id, depth="standard")
        except Exception:
            # Inline fallback
            from services.summary_service import summary_service
            from utils.rag_engine import rag_engine
            import asyncio
            asyncio.create_task(rag_engine.index_document_chunks(merged_id, final_extracted_text))
            asyncio.create_task(summary_service.generate_summary_for_text(merged_id, final_extracted_text, depth="standard"))

        return merged_doc_record

    async def list_documents_paginated(
        self, user_id: str, page: int = 1, limit: int = 10
    ) -> Dict[str, Any]:
        """Returns paginated list of user's documents."""
        offset = (page - 1) * limit
        items = []
        total = 0

        try:
            res = supabase.table("documents").select("*", count="exact").eq("user_id", user_id).range(offset, offset + limit - 1).execute()
            items = res.data or []
            total = res.count or len(items)
        except Exception:
            pass

        if not items:
            user_docs = [d for d in _in_memory_docs.values() if d.get("user_id") == user_id]
            total = len(user_docs)
            items = user_docs[offset:offset + limit]

        return {
            "items": items,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": (total + limit - 1) // limit if limit > 0 else 1
        }

    async def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Lists user's documents."""
        try:
            res = supabase.table("documents").select("*").eq("user_id", user_id).execute()
            if res.data:
                return res.data
        except Exception:
            pass

        return [d for d in _in_memory_docs.values() if d.get("user_id") == user_id]

    async def delete_document(self, doc_id: str, user_id: str) -> bool:
        """Deletes a document record and its associated storage file."""
        doc = await self.get_document(doc_id, user_id)
        if not doc:
            return False

        file_url = doc.get("file_url", "")
        if file_url:
            try:
                storage_path = file_url.split("/documents/")[-1] if "/documents/" in file_url else file_url
                supabase.storage.from_("documents").remove([storage_path])
            except Exception:
                pass

        if doc_id in _in_memory_docs:
            del _in_memory_docs[doc_id]

        try:
            supabase.table("documents").delete().eq("id", doc_id).eq("user_id", user_id).execute()
        except Exception:
            pass

        return True

document_service = DocumentService()
