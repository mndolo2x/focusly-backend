from datetime import datetime, timedelta
import uuid
from typing import Any, Dict, List, Optional
from database import supabase
from services.ollama_service import ollama_service

class StudyService:
    """Service for spaced repetition scheduling and custom study plan creation."""

    async def generate_study_plan(self, user_id: str, exam_date: datetime, document_ids: List[str]) -> Dict[str, Any]:
        """Generates a study schedule up to the exam date."""
        days_until_exam = (exam_date - datetime.utcnow()).days
        if days_until_exam <= 0:
            days_until_exam = 7

        prompt = (
            f"Generate a daily study schedule for a high school student with an exam on {exam_date.strftime('%Y-%m-%d')}.\n"
            f"Days remaining: {days_until_exam}.\n"
            f"Return a JSON object with key 'schedule' as a list of daily plans with 'day_number', 'topics_to_review', and 'estimated_minutes'."
        )
        system = "You are an academic study planner helping high school students prepare for exams effectively."

        try:
            data = await ollama_service.generate_json(prompt, system_prompt=system)
            schedule = data.get("schedule", [])
        except Exception:
            schedule = []

        if not schedule:
            schedule = [
                {"day_number": i + 1, "topics_to_review": ["Review chapter materials and quizzes"], "estimated_minutes": 30}
                for i in range(min(days_until_exam, 14))
            ]

        record = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "exam_date": exam_date.isoformat(),
            "schedule": schedule
        }

        try:
            res = supabase.table("study_plans").insert(record).execute()
            if res.data:
                return res.data[0]
        except Exception:
            pass

        return record

study_service = StudyService()
