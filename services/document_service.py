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
        self, user_id: str, filename: str, file_url: Optional[str] = None, page_count: int = 0
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
        if doc and doc.get("user_id") == user_id:
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
