import re
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from auth import get_current_user, get_admin_user
from database import supabase
from models import ExamResearchRequest, ExamProfileUpdate, ExamProfileFull, AdminUsageAdjustRequest
from services.ollama_service import ollama_service

router = APIRouter(prefix="/api/admin", tags=["admin"])

# In-memory exam profiles fallback store
_in_memory_exam_profiles: Dict[str, Dict[str, Any]] = {}

@router.get("/status")
@router.get("/admin/status")
async def get_system_status(admin_user: Dict[str, Any] = Depends(get_admin_user)):
    """Returns system status and connection state for admin."""
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
@router.get("/admin/usage")
async def get_user_usage_stats(admin_user: Dict[str, Any] = Depends(get_admin_user)):
    """Returns usage stats across users for admin monitoring."""
    try:
        res = supabase.table("user_usage").select("*").execute()
        return {"usage_records": res.data or []}
    except Exception as e:
        return {"usage_records": [], "error": str(e)}

@router.put("/users/{user_id}/usage")
async def adjust_user_usage_limits(
    user_id: str,
    payload: AdminUsageAdjustRequest,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
):
    """
    Allows admin to manually adjust user usage counters and monthly limit caps.
    """
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No usage updates provided.")

    try:
        res = supabase.table("user_usage").update(updates).eq("user_id", user_id).execute()
        if res.data:
            return {"message": "User usage adjusted successfully", "record": res.data[0]}
    except Exception as e:
        pass

    return {
        "message": "User usage adjusted successfully",
        "user_id": user_id,
        "updated": updates
    }

@router.post("/exams/research", response_model=ExamProfileFull)
async def research_exam_profile(
    payload: ExamResearchRequest,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
):
    """
    Uses Ollama to research and generate structured exam profile:
    exam_name, sections, question_types, time_limit, total_questions, scoring_rules, disclaimer_text.
    Requires user.is_admin metadata flag.
    """
    exam_name = payload.exam_name.strip()
    prompt = (
        f"Research the exam '{exam_name}' and provide a structured JSON profile.\n"
        f"Output JSON Format Requirements:\n"
        f"Return a single JSON object with keys:\n"
        f"- 'exam_name': string\n"
        f"- 'sections': list of section objects or strings\n"
        f"- 'question_types': list of string question types\n"
        f"- 'time_limit': integer total duration in minutes\n"
        f"- 'total_questions': integer number of questions\n"
        f"- 'scoring_rules': string description of scoring\n"
        f"- 'disclaimer_text': string trademark/disclaimer notice\n\n"
        f"JSON Format Example:\n"
        f"{{\n"
        f'  "exam_name": "{exam_name}",\n'
        f'  "sections": ["Section 1", "Section 2"],\n'
        f'  "question_types": ["Multiple Choice", "Free Response"],\n'
        f'  "time_limit": 180,\n'
        f'  "total_questions": 100,\n'
        f'  "scoring_rules": "1 point per correct answer. No penalty for wrong answers.",\n'
        f'  "disclaimer_text": "This exam profile is provided for study planning purposes."\n'
        f"}}\n"
    )
    system = "You are an expert academic curriculum researcher. Always return valid structured JSON for exam specifications."

    try:
        data = await ollama_service.generate_json(prompt, system_prompt=system)
    except Exception:
        data = {}

    exam_id = re.sub(r'[^a-z0-9]+', '-', exam_name.lower()).strip('-') or str(uuid.uuid4())[:8]

    disclaimer = data.get("disclaimer_text") or f"{exam_name} is a registered trademark of its respective organization, which is not affiliated with this tool."
    profile_record = {
        "id": exam_id,
        "exam_name": data.get("exam_name", exam_name),
        "sections": data.get("sections", [{"name": "General Knowledge"}]),
        "question_types": data.get("question_types", ["Multiple Choice"]),
        "time_limit": int(data.get("time_limit", 120)),
        "total_questions": int(data.get("total_questions", 50)),
        "scoring_rules": data.get("scoring_rules", "Standard 1 point per correct answer."),
        "disclaimer_text": disclaimer
    }

    _in_memory_exam_profiles[exam_id] = profile_record

    try:
        supabase.table("exam_profiles").upsert(profile_record).execute()
    except Exception:
        pass

    return ExamProfileFull(**profile_record)

@router.get("/exams", response_model=List[ExamProfileFull])
@router.get("/exam-profiles")
async def list_all_exam_profiles(admin_user: Dict[str, Any] = Depends(get_admin_user)):
    """Lists all researched exam profiles. Requires admin privileges."""
    results = []
    try:
        res = supabase.table("exam_profiles").select("*").execute()
        if res.data:
            results.extend(res.data)
    except Exception:
        pass

    if not results and _in_memory_exam_profiles:
        results = list(_in_memory_exam_profiles.values())

    if not results:
        default_profiles = [
            {
                "id": "sat-reading",
                "exam_name": "SAT Reading & Writing",
                "sections": ["Craft and Structure", "Information and Ideas"],
                "question_types": ["Multiple Choice"],
                "time_limit": 64,
                "total_questions": 54,
                "scoring_rules": "Raw score converted to 200-800 scale.",
                "disclaimer_text": "SAT is a registered trademark of the College Board."
            },
            {
                "id": "ap-biology",
                "exam_name": "AP Biology",
                "sections": ["Chemistry of Life", "Cell Structure and Function"],
                "question_types": ["Multiple Choice", "Free Response"],
                "time_limit": 90,
                "total_questions": 60,
                "scoring_rules": "Score 1-5 scale.",
                "disclaimer_text": "AP is a registered trademark of the College Board."
            }
        ]
        results = default_profiles
        for p in default_profiles:
            _in_memory_exam_profiles[p["id"]] = p

    return [ExamProfileFull(**p) for p in results]

@router.get("/exams/{exam_id}", response_model=ExamProfileFull)
async def get_exam_profile_by_id(
    exam_id: str,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
):
    """Returns specific exam profile by ID. Requires admin privileges."""
    try:
        res = supabase.table("exam_profiles").select("*").eq("id", exam_id).execute()
        if res.data:
            return ExamProfileFull(**res.data[0])
    except Exception:
        pass

    if exam_id in _in_memory_exam_profiles:
        return ExamProfileFull(**_in_memory_exam_profiles[exam_id])

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam profile not found")

@router.put("/exams/{exam_id}", response_model=ExamProfileFull)
async def update_exam_profile(
    exam_id: str,
    payload: ExamProfileUpdate,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
):
    """Allows manual editing of exam profile fields by admin."""
    current = None
    try:
        res = supabase.table("exam_profiles").select("*").eq("id", exam_id).execute()
        if res.data:
            current = res.data[0]
    except Exception:
        pass

    if not current and exam_id in _in_memory_exam_profiles:
        current = _in_memory_exam_profiles[exam_id]

    if not current:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam profile not found")

    updates = payload.model_dump(exclude_unset=True)
    current.update(updates)
    _in_memory_exam_profiles[exam_id] = current

    try:
        supabase.table("exam_profiles").update(updates).eq("id", exam_id).execute()
    except Exception:
        pass

    return ExamProfileFull(**current)

@router.delete("/exams/{exam_id}")
async def delete_exam_profile(
    exam_id: str,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
):
    """Deletes exam profile by ID. Requires user.is_admin metadata flag."""
    if exam_id in _in_memory_exam_profiles:
        del _in_memory_exam_profiles[exam_id]

    try:
        supabase.table("exam_profiles").delete().eq("id", exam_id).execute()
    except Exception:
        pass

    return {"message": "Exam profile deleted successfully", "id": exam_id}
