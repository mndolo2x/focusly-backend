import uuid
from typing import Any, Dict, List, Optional
from database import supabase
from models import DocumentStatus

class DocumentService:
    """Service for managing document uploads, database state, and text processing."""

    async def create_document_record(self, user_id: str, filename: str, file_url: Optional[str] = None) -> Dict[str, Any]:
        """Creates a document database entry with status 'processing'."""
        doc_id = str(uuid.uuid4())
        record = {
            "id": doc_id,
            "user_id": user_id,
            "filename": filename,
            "file_url": file_url or "",
            "extracted_text": "",
            "status": DocumentStatus.PROCESSING,
            "page_count": 0,
        }
        try:
            res = supabase.table("documents").insert(record).execute()
            if res.data:
                return res.data[0]
        except Exception:
            pass
        return record

    async def update_document(self, doc_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Updates document record in Supabase."""
        try:
            res = supabase.table("documents").update(updates).eq("id", doc_id).execute()
            if res.data:
                return res.data[0]
        except Exception:
            pass
        return {"id": doc_id, **updates}

    async def get_document(self, doc_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves user's document record."""
        try:
            res = supabase.table("documents").select("*").eq("id", doc_id).eq("user_id", user_id).execute()
            if res.data:
                return res.data[0]
        except Exception:
            pass
        return None

    async def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Lists user's documents."""
        try:
            res = supabase.table("documents").select("*").eq("user_id", user_id).execute()
            if res.data:
                return res.data
        except Exception:
            pass
        return []

document_service = DocumentService()
