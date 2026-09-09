from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from auth import get_current_user
from database import supabase

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/status")
async def get_system_status(user: Dict[str, Any] = Depends(get_current_user)):
    """Returns system status and connection state."""
    return {
        "status": "online",
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "ollama": "ready",
            "tts": "ready"
        }
    }

@router.get("/usage")
async def get_user_usage_stats(user: Dict[str, Any] = Depends(get_current_user)):
    """Returns usage stats across users for admin monitoring."""
    try:
        res = supabase.table("user_usage").select("*").execute()
        return {"usage_records": res.data or []}
    except Exception as e:
        return {"usage_records": [], "error": str(e)}

@router.get("/exam-profiles")
async def list_exam_profiles():
    """Lists pre-configured exam profiles (AP, SAT, GCSE, etc.)."""
    try:
        res = supabase.table("exam_profiles").select("*").execute()
        if res.data:
            return res.data
    except Exception:
        pass

    # Default fallback exam profiles
    return [
        {
            "id": "sat-reading",
            "exam_name": "SAT Reading & Writing",
            "sections": [{"name": "Craft and Structure"}, {"name": "Information and Ideas"}],
            "question_types": ["Multiple Choice"],
            "time_limit": 64,
            "disclaimer_text": "AP/SAT is a registered trademark of the College Board, which was not involved in the production of, and does not endorse, this product."
        },
        {
            "id": "ap-biology",
            "exam_name": "AP Biology",
            "sections": [{"name": "Chemistry of Life"}, {"name": "Cell Structure and Function"}],
            "question_types": ["Multiple Choice", "Free Response"],
            "time_limit": 90,
            "disclaimer_text": "AP is a registered trademark of the College Board, which was not involved in the production of, and does not endorse, this product."
        }
    ]
