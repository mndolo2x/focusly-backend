from datetime import datetime, timedelta
import uuid
from typing import Any, Dict, List, Optional
from database import supabase
from services.ollama_service import ollama_service
from services.document_service import document_service
from models import EXAM_PAPER_DISCLAIMER

# In-memory exam papers fallback store
_in_memory_exam_papers: Dict[str, Dict[str, Any]] = {}

class StudyService:
    """Service for spaced repetition scheduling, custom study plans, and exam paper generation."""

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

    async def generate_exam_paper(
        self, user_id: str, exam_id: str, document_ids: List[str], num_questions: int = 25
    ) -> Dict[str, Any]:
        """
        Uses Ollama to generate an exam paper matching the exact format of the target exam profile.
        CRITICAL: Includes mandatory disclaimer in response:
        "This is an unofficial practice paper and is not affiliated with or endorsed by the exam board."
        """
        # Retrieve exam profile details
        exam_profile = {"exam_name": "Practice Exam", "time_limit": 60}
        try:
            res_ex = supabase.table("exam_profiles").select("*").eq("id", exam_id).execute()
            if res_ex.data:
                exam_profile = res_ex.data[0]
        except Exception:
            pass

        # Retrieve text from source documents
        combined_text = ""
        for d_id in document_ids:
            doc = await document_service.get_document(d_id, user_id)
            if doc and doc.get("extracted_text"):
                combined_text += f"\n--- Document {d_id} ---\n" + doc["extracted_text"][:4000]

        exam_name = exam_profile.get("exam_name", "Practice Exam")
        time_limit = exam_profile.get("time_limit", 60)

        prompt = (
            f"Generate a formal mock exam paper for '{exam_name}' with exactly {num_questions} multiple choice questions.\n"
            f"Matching Exam Profile Format:\n"
            f"- Question types: {exam_profile.get('question_types', ['Multiple Choice'])}\n"
            f"- Sections: {exam_profile.get('sections', ['General'])}\n"
            f"Requirements:\n"
            f"- Each question MUST have 4 options.\n"
            f"- 'correct_answer_index' must be integer 0 to 3.\n"
            f"- Format output as JSON object with key 'questions' containing list of objects:\n"
            f"  [{{\"id\": \"q1\", \"question_text\": \"...\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"correct_answer_index\": 0}}]\n\n"
            f"Source Content:\n{combined_text[:8000] if combined_text else 'General curriculum concepts.'}"
        )
        system = f"You are an expert exam board question writer for {exam_name}. Create accurate, challenging practice questions."

        try:
            data = await ollama_service.generate_json(prompt, system_prompt=system)
            raw_qs = data.get("questions", [])
        except Exception:
            raw_qs = []

        if not raw_qs or len(raw_qs) < num_questions:
            fallback_qs = []
            for i in range(num_questions):
                q_id = f"q_{uuid.uuid4().hex[:8]}"
                fallback_qs.append({
                    "id": q_id,
                    "question_text": f"Exam Question {i+1} regarding {exam_name} key concepts:",
                    "options": [
                        f"Correct answer choice for Q{i+1}",
                        f"Incorrect choice A for Q{i+1}",
                        f"Incorrect choice B for Q{i+1}",
                        f"Incorrect choice C for Q{i+1}"
                    ],
                    "correct_answer_index": 0
                })
            raw_qs = fallback_qs

        paper_id = str(uuid.uuid4())
        paper_record = {
            "id": paper_id,
            "user_id": user_id,
            "exam_id": exam_id,
            "exam_name": exam_name,
            "document_ids": document_ids,
            "questions": raw_qs[:num_questions],
            "time_limit": time_limit,
            "total_questions": len(raw_qs[:num_questions]),
            "disclaimer_text": EXAM_PAPER_DISCLAIMER,
            "created_at": datetime.utcnow().isoformat()
        }

        _in_memory_exam_papers[paper_id] = paper_record
        for d_id in document_ids:
            _in_memory_exam_papers[f"doc_{d_id}"] = paper_record

        return paper_record

    async def get_document_exam_paper(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Returns the most recent generated exam paper for a document."""
        key = f"doc_{document_id}"
        if key in _in_memory_exam_papers:
            return _in_memory_exam_papers[key]

        for paper in _in_memory_exam_papers.values():
            if document_id in paper.get("document_ids", []):
                return paper

        return None

    async def submit_exam_paper(self, user_id: str, paper_id: str, submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Grades submitted exam paper and returns score breakdown with mandatory disclaimer."""
        paper = _in_memory_exam_papers.get(paper_id)
        if not paper:
            # Fallback mock paper
            paper = {
                "id": paper_id,
                "exam_id": "mock_exam",
                "exam_name": "Practice Exam",
                "questions": [],
                "disclaimer_text": EXAM_PAPER_DISCLAIMER
            }

        questions = paper.get("questions", [])
        q_map = {q.get("id"): q for q in questions if isinstance(q, dict)}

        total_questions = len(submissions) or len(questions) or 1
        correct_count = 0
        incorrect_count = 0
        breakdown = []

        for sub in submissions:
            q_id = sub.get("question_id")
            selected_ans = sub.get("selected_answer", 0)
            q = q_map.get(q_id, {})
            correct_ans = q.get("correct_answer_index", 0)

            is_correct = (selected_ans == correct_ans)
            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1

            breakdown.append({
                "question_id": q_id,
                "question_text": q.get("question_text", ""),
                "selected_answer": selected_ans,
                "correct_answer": correct_ans,
                "is_correct": is_correct
            })

        score_pct = round((correct_count / total_questions) * 100, 2)

        return {
            "paper_id": paper_id,
            "exam_id": paper.get("exam_id", "mock_exam"),
            "exam_name": paper.get("exam_name", "Practice Exam"),
            "score": score_pct,
            "total_questions": total_questions,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "breakdown": breakdown,
            "disclaimer_text": EXAM_PAPER_DISCLAIMER
        }

study_service = StudyService()
