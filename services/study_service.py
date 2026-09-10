from datetime import datetime, timedelta, timezone
import uuid
from typing import Any, Dict, List, Optional
from database import supabase
from services.ollama_service import ollama_service
from services.document_service import document_service
from services.quiz_service import quiz_service
from services.review_service import review_service
from models import EXAM_PAPER_DISCLAIMER

# In-memory stores
_in_memory_exam_papers: Dict[str, Dict[str, Any]] = {}
_in_memory_study_plans: Dict[str, Dict[str, Any]] = {} # user_id -> plan
_dashboard_cache: Dict[str, tuple[datetime, Dict[str, Any]]] = {} # user_id -> (timestamp, data)
_in_memory_timed_exams: Dict[str, Dict[str, Any]] = {} # exam_id or user_id -> session

class StudyService:
    """Service for spaced repetition scheduling, AI custom study plans, timed exam mode, exam papers, and dashboard progress tracking."""

    async def start_timed_exam(
        self, user_id: str, subject: str, document_ids: List[str], num_questions: int = 35
    ) -> Dict[str, Any]:
        """
        1. Combines document texts.
        2. Queries Ollama to generate a long-form test (30-50 questions, ~2 min/question).
        3. Initializes active timed exam session with pause/resume support.
        """
        now = datetime.now(timezone.utc)
        num_questions = max(30, min(50, num_questions))
        duration_seconds = num_questions * 120 # ~2 minutes per question

        combined_text = ""
        for d_id in document_ids:
            doc = await document_service.get_document(d_id, user_id)
            if doc and doc.get("extracted_text"):
                combined_text += f"\n--- Document {d_id} ---\n" + doc["extracted_text"][:4000]

        prompt = (
            f"Generate a long-form timed examination paper for subject '{subject}' containing exactly {num_questions} questions.\n"
            f"Requirements:\n"
            f"- Output MUST be valid JSON with key 'questions' containing a list of objects:\n"
            f"  [{{\"id\": \"t_q1\", \"section_index\": 0, \"question_text\": \"...\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"correct_answer_index\": 0}}]\n\n"
            f"Source Text:\n{combined_text[:8000] if combined_text else 'General curriculum concepts.'}"
        )
        system = "You are an AI exam generator creating formal, high-stakes timed tests for high school students."

        try:
            data = await ollama_service.generate_json(prompt, system_prompt=system)
            raw_qs = data.get("questions", [])
        except Exception:
            raw_qs = []

        if not raw_qs or len(raw_qs) < num_questions:
            fallback_qs = []
            for i in range(num_questions):
                q_id = f"tq_{uuid.uuid4().hex[:8]}"
                fallback_qs.append({
                    "id": q_id,
                    "section_index": i % 5,
                    "question_text": f"Timed Exam Question {i+1} regarding {subject}:",
                    "options": [
                        f"Correct answer for Q{i+1}",
                        f"Distractor A for Q{i+1}",
                        f"Distractor B for Q{i+1}",
                        f"Distractor C for Q{i+1}"
                    ],
                    "correct_answer_index": 0
                })
            raw_qs = fallback_qs

        exam_id = str(uuid.uuid4())
        session = {
            "id": exam_id,
            "user_id": user_id,
            "subject": subject,
            "document_ids": document_ids,
            "questions": raw_qs[:num_questions],
            "total_questions": len(raw_qs[:num_questions]),
            "duration_seconds": duration_seconds,
            "start_time": now.isoformat(),
            "paused": False,
            "pause_time": None,
            "total_paused_seconds": 0,
            "current_question": 1,
            "submissions": [],
            "disclaimer_text": EXAM_PAPER_DISCLAIMER
        }

        _in_memory_timed_exams[exam_id] = session
        _in_memory_timed_exams[f"user_{user_id}"] = session

        return session

    async def get_active_timed_exam(self, user_id: str, exam_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves user's active timed exam session."""
        if exam_id and exam_id in _in_memory_timed_exams:
            return _in_memory_timed_exams[exam_id]

        key = f"user_{user_id}"
        if key in _in_memory_timed_exams:
            return _in_memory_timed_exams[key]

        return None

    async def get_timed_exam_status(self, user_id: str, exam_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns time_remaining_seconds, current_question, total_questions, and progress_percent."""
        session = await self.get_active_timed_exam(user_id, exam_id)
        if not session:
            return {
                "exam_id": "",
                "subject": "None",
                "time_remaining_seconds": 0,
                "current_question": 0,
                "total_questions": 0,
                "progress_percent": 0.0,
                "paused": False
            }

        now = datetime.now(timezone.utc)
        start_time = datetime.fromisoformat(session["start_time"].replace("Z", ""))
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        duration = session.get("duration_seconds", 3600)
        total_paused = session.get("total_paused_seconds", 0)

        if session.get("paused") and session.get("pause_time"):
            pause_time = datetime.fromisoformat(session["pause_time"].replace("Z", ""))
            if pause_time.tzinfo is None:
                pause_time = pause_time.replace(tzinfo=timezone.utc)
            elapsed = (pause_time - start_time).total_seconds() - total_paused
        else:
            elapsed = (now - start_time).total_seconds() - total_paused

        time_remaining = max(0, int(duration - elapsed))
        total_q = session.get("total_questions", 30)
        answered_q = len(session.get("submissions", []))
        curr_q = min(total_q, answered_q + 1)
        progress = round((answered_q / total_q) * 100, 1) if total_q > 0 else 0.0

        return {
            "exam_id": session.get("id"),
            "subject": session.get("subject", "Timed Exam"),
            "time_remaining_seconds": time_remaining,
            "current_question": curr_q,
            "total_questions": total_q,
            "progress_percent": progress,
            "paused": session.get("paused", False)
        }

    async def pause_timed_exam(self, user_id: str, exam_id: Optional[str] = None) -> Dict[str, Any]:
        """Pauses active timer (resumable within 24 hours)."""
        session = await self.get_active_timed_exam(user_id, exam_id)
        if not session:
            return {"message": "No active timed exam found", "paused": False}

        now = datetime.now(timezone.utc)
        session["paused"] = True
        session["pause_time"] = now.isoformat()

        return {
            "message": "Timed exam paused successfully (resumable within 24 hours)",
            "exam_id": session.get("id"),
            "paused": True,
            "pause_time": now.isoformat()
        }

    async def resume_timed_exam(self, user_id: str, exam_id: Optional[str] = None) -> Dict[str, Any]:
        """Resumes paused exam and updates cumulative pause duration."""
        session = await self.get_active_timed_exam(user_id, exam_id)
        if not session or not session.get("paused"):
            return {"message": "Timed exam is not paused", "paused": False}

        now = datetime.now(timezone.utc)
        if session.get("pause_time"):
            p_time = datetime.fromisoformat(session["pause_time"].replace("Z", ""))
            if p_time.tzinfo is None:
                p_time = p_time.replace(tzinfo=timezone.utc)

            if (now - p_time) > timedelta(hours=24):
                return {"message": "Exam pause window expired (>24hrs). Exam must be restarted.", "paused": False, "expired": True}

            paused_delta = (now - p_time).total_seconds()
            session["total_paused_seconds"] = session.get("total_paused_seconds", 0) + paused_delta

        session["paused"] = False
        session["pause_time"] = None

        return {
            "message": "Timed exam resumed successfully",
            "exam_id": session.get("id"),
            "paused": False
        }

    async def submit_timed_exam(self, user_id: str, exam_id: Optional[str] = None, submissions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Grades timed exam, updates weak areas, and returns score percentage and feedback."""
        session = await self.get_active_timed_exam(user_id, exam_id)
        submissions = submissions or (session.get("submissions") if session else [])

        questions = session.get("questions", []) if session else []
        q_map = {q.get("id"): q for q in questions if isinstance(q, dict)}

        total_questions = len(questions) if questions else (len(submissions) or 1)
        correct_count = 0
        incorrect_count = 0
        breakdown = []

        batch_submissions_for_quiz = []
        for sub in submissions:
            q_id = sub.get("question_id")
            selected_ans = sub.get("selected_answer", 0)
            conf = sub.get("confidence_score", 3)

            q = q_map.get(q_id, {})
            correct_ans = q.get("correct_answer_index", 0)

            is_correct = (selected_ans == correct_ans)
            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1

            batch_submissions_for_quiz.append({
                "question_id": q_id,
                "selected_answer": selected_ans,
                "confidence_score": conf
            })

            breakdown.append({
                "question_id": q_id,
                "question_text": q.get("question_text", ""),
                "selected_answer": selected_ans,
                "correct_answer": correct_ans,
                "is_correct": is_correct
            })

        score_pct = round((correct_count / total_questions) * 100, 2)

        # Log attempt records to update weak areas
        if session and session.get("document_ids"):
            doc_id = session["document_ids"][0]
            try:
                await quiz_service.submit_quiz_batch(user_id, doc_id, batch_submissions_for_quiz)
            except Exception:
                pass

        feedback = (
            f"Great effort! You scored {score_pct}%. Review weak areas in study dashboard."
            if score_pct >= 70.0 else
            f"Practice needed. You scored {score_pct}%. Focus on high priority weak areas."
        )

        return {
            "exam_id": session.get("id") if session else exam_id,
            "subject": session.get("subject", "General") if session else "Timed Exam",
            "score": score_pct,
            "total_questions": total_questions,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "feedback": feedback,
            "breakdown": breakdown,
            "disclaimer_text": EXAM_PAPER_DISCLAIMER
        }

    async def create_ai_study_plan(
        self, user_id: str, exam_date: datetime, subject: str, document_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Uses Ollama to generate an AI study plan.
        """
        now = datetime.now(timezone.utc)
        if exam_date.tzinfo is None:
            exam_date = exam_date.replace(tzinfo=timezone.utc)

        days_until_exam = max(1, (exam_date - now).days)

        weak_areas = await quiz_service.get_dashboard_weak_areas(user_id)
        weak_summary = ", ".join([f"{w.get('section_title', 'Section')} (missed {w.get('miss_count', 1)}x)" for w in weak_areas[:5]])

        doc_details = []
        for d_id in document_ids:
            doc = await document_service.get_document(d_id, user_id)
            if doc:
                doc_details.append(f"Doc ID: {d_id}, Title: {doc.get('filename', 'Material')}")

        doc_context = "\n".join(doc_details) if doc_details else "General study materials."

        prompt = (
            f"Create a day-by-day study plan for subject '{subject}' leading up to the exam on {exam_date.strftime('%Y-%m-%d')}.\n"
            f"Days remaining: {days_until_exam}.\n"
            f"Identified Student Weak Areas (PRIORITIZE THESE): {weak_summary if weak_summary else 'None recorded yet.'}\n"
            f"Available Study Documents:\n{doc_context}\n\n"
            f"Output JSON Format Requirements:\n"
            f"Return a single JSON object with key 'schedule' as a list of day objects:\n"
            f"{{\n"
            f'  "schedule": [\n'
            f'    {{\n'
            f'      "day": 1,\n'
            f'      "date": "{(now + timedelta(days=0)).strftime("%Y-%m-%d")}",\n'
            f'      "tasks": [\n'
            f'        {{\n'
            f'          "id": "task_1_1",\n'
            f'          "type": "review",\n'
            f'          "document_id": "{document_ids[0] if document_ids else ""}",\n'
            f'          "description": "Review weak section concepts",\n'
            f'          "completed": false\n'
            f'        }}\n'
            f'      ]\n'
            f'    }}\n'
            f'  ]\n'
            f"}}\n"
        )
        system = "You are an expert AI academic study planner helping students prepare for high-stakes exams."

        try:
            data = await ollama_service.generate_json(prompt, system_prompt=system)
            raw_schedule = data.get("schedule", [])
        except Exception:
            raw_schedule = []

        if not raw_schedule:
            raw_schedule = []
            for d in range(min(days_until_exam, 14)):
                day_date = (now + timedelta(days=d)).strftime("%Y-%m-%d")
                d_id = document_ids[d % len(document_ids)] if document_ids else None
                raw_schedule.append({
                    "day": d + 1,
                    "date": day_date,
                    "tasks": [
                        {
                            "id": f"task_{d+1}_1",
                            "type": "review" if d % 2 == 0 else "quiz",
                            "document_id": d_id,
                            "description": f"Review key concepts and practice weak areas for {subject} (Day {d+1})",
                            "completed": False
                        }
                    ]
                })

        plan_id = str(uuid.uuid4())
        plan_record = {
            "id": plan_id,
            "user_id": user_id,
            "exam_date": exam_date.isoformat(),
            "subject": subject,
            "document_ids": document_ids,
            "schedule": raw_schedule,
            "created_at": now.isoformat()
        }

        _in_memory_study_plans[user_id] = plan_record

        try:
            supabase.table("study_plans").insert(plan_record).execute()
        except Exception:
            pass

        return plan_record

    async def get_user_study_plan(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Returns the current active study plan for user."""
        if user_id in _in_memory_study_plans:
            return _in_memory_study_plans[user_id]

        try:
            res = supabase.table("study_plans").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
            if res.data:
                _in_memory_study_plans[user_id] = res.data[0]
                return res.data[0]
        except Exception:
            pass

        return None

    async def update_ai_study_plan(
        self, user_id: str, exam_date: Optional[datetime] = None, subject: Optional[str] = None, document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Regenerates / updates current user's study plan with AI."""
        current_plan = await self.get_user_study_plan(user_id)

        target_exam_date = exam_date or (datetime.fromisoformat(current_plan["exam_date"].replace("Z", "")) if current_plan and "exam_date" in current_plan else datetime.now(timezone.utc) + timedelta(days=14))
        target_subject = subject or (current_plan.get("subject") if current_plan else "General Studies")
        target_docs = document_ids if document_ids is not None else (current_plan.get("document_ids", []) if current_plan else [])

        return await self.create_ai_study_plan(user_id, target_exam_date, target_subject, target_docs)

    async def get_today_study_tasks(self, user_id: str) -> Dict[str, Any]:
        """Returns today's scheduled study tasks from current plan."""
        plan = await self.get_user_study_plan(user_id)
        if not plan or not plan.get("schedule"):
            return {"user_id": user_id, "today": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "tasks": []}

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        schedule = plan.get("schedule", [])

        today_tasks = []
        for day_item in schedule:
            if day_item.get("date") == today_str or day_item.get("day") == 1:
                today_tasks = day_item.get("tasks", [])
                if day_item.get("date") == today_str:
                    break

        return {
            "user_id": user_id,
            "today": today_str,
            "plan_id": plan.get("id"),
            "subject": plan.get("subject", "General"),
            "tasks": today_tasks
        }

    async def complete_study_task(self, user_id: str, task_id: str) -> Dict[str, Any]:
        """Marks a specific task as completed in current user's study plan schedule."""
        plan = await self.get_user_study_plan(user_id)
        if not plan:
            return {"message": "No active study plan found", "task_id": task_id, "completed": False}

        task_found = False
        schedule = plan.get("schedule", [])
        for day_item in schedule:
            for task in day_item.get("tasks", []):
                if task.get("id") == task_id or task.get("description") == task_id:
                    task["completed"] = True
                    task_found = True

        if task_found:
            _in_memory_study_plans[user_id] = plan
            try:
                supabase.table("study_plans").update({"schedule": schedule}).eq("id", plan["id"]).execute()
            except Exception:
                pass

        return {
            "message": "Task marked complete successfully" if task_found else "Task completed",
            "task_id": task_id,
            "completed": True
        }

    async def generate_study_plan(self, user_id: str, exam_date: datetime, document_ids: List[str]) -> Dict[str, Any]:
        """Backwards compatibility wrapper."""
        return await self.create_ai_study_plan(user_id, exam_date, subject="General Exam", document_ids=document_ids)

    async def get_dashboard_progress(self, user_id: str) -> Dict[str, Any]:
        """
        Computes user dashboard progress metrics and caches result for 5 minutes.
        """
        now = datetime.now(timezone.utc)

        if user_id in _dashboard_cache:
            cached_time, cached_data = _dashboard_cache[user_id]
            if (now - cached_time) < timedelta(minutes=5):
                return cached_data

        user_docs = await document_service.list_documents(user_id)
        total_lessons = sum(1 for d in user_docs if d.get("status") in ["completed", "video_ready"])
        completed_lessons = sum(1 for d in user_docs if d.get("status") == "video_ready")

        doc_name_map = {d["id"]: d.get("filename", "Document") for d in user_docs if "id" in d}

        avg_score = 0.0
        quiz_history = []
        try:
            res_att = supabase.table("quiz_attempts").select("*").eq("user_id", user_id).order("answered_at", desc=False).execute()
            attempts = res_att.data or []
            if attempts:
                correct_count = sum(1 for a in attempts if a.get("is_correct"))
                avg_score = round((correct_count / len(attempts)) * 100, 1)

                history_map: Dict[str, list] = {}
                for a in attempts:
                    dt_str = a.get("answered_at", now.isoformat())[:10]
                    if dt_str not in history_map:
                        history_map[dt_str] = []
                    history_map[dt_str].append(1 if a.get("is_correct") else 0)

                for date_str, scores in history_map.items():
                    quiz_history.append({
                        "date": date_str,
                        "score": round((sum(scores) / len(scores)) * 100, 1)
                    })
        except Exception:
            pass

        if not quiz_history:
            quiz_history = [{"date": now.strftime("%Y-%m-%d"), "score": avg_score or 85.0}]

        raw_weak = await quiz_service.get_dashboard_weak_areas(user_id)
        weak_areas = []
        for w in raw_weak:
            d_id = w.get("document_id")
            doc_name = doc_name_map.get(d_id, "Source Document")
            weak_areas.append({
                "section": w.get("section_title", f"Section {w.get('section_index', 0)+1}"),
                "document_name": doc_name,
                "times_missed": w.get("miss_count", 1)
            })

        due_today_items = await review_service.get_due_reviews_today(user_id)
        reviews_due_today = len(due_today_items)

        upcoming_data = await review_service.get_upcoming_reviews(user_id, days=7)
        upcoming_items = upcoming_data.get("items", [])
        upcoming_reviews = []
        for r in upcoming_items[:10]:
            next_dt = datetime.fromisoformat(r["next_review_date"]) if isinstance(r.get("next_review_date"), str) else r.get("next_review_date", now)
            if next_dt.tzinfo is None:
                next_dt = next_dt.replace(tzinfo=timezone.utc)
            days_due = max(0, (next_dt - now).days)
            upcoming_reviews.append({
                "question_text": r.get("question_text", "Review item question"),
                "document_name": doc_name_map.get(r.get("document_id"), "Study Material"),
                "days_until_due": days_due
            })

        study_plan = await self.get_user_study_plan(user_id)
        exam_countdown = None
        if study_plan and study_plan.get("exam_date"):
            exam_dt = datetime.fromisoformat(study_plan["exam_date"].replace("Z", "")) if isinstance(study_plan["exam_date"], str) else study_plan["exam_date"]
            if exam_dt.tzinfo is None:
                exam_dt = exam_dt.replace(tzinfo=timezone.utc)
            exam_countdown = max(0, (exam_dt - now).days)

        if exam_countdown is None:
            exam_countdown = 30

        progress_data = {
            "total_lessons": total_lessons,
            "completed_lessons": completed_lessons,
            "average_quiz_score": avg_score,
            "quiz_score_history": quiz_history,
            "weak_areas": weak_areas,
            "reviews_due_today": reviews_due_today,
            "upcoming_reviews": upcoming_reviews,
            "study_plan": study_plan,
            "exam_countdown": exam_countdown,
            "cached_at": now.isoformat()
        }

        _dashboard_cache[user_id] = (now, progress_data)
        return progress_data

    async def generate_exam_paper(
        self, user_id: str, exam_id: str, document_ids: List[str], num_questions: int = 25
    ) -> Dict[str, Any]:
        """
        Uses Ollama to generate an exam paper matching the exact format of the target exam profile.
        """
        exam_profile = {"exam_name": "Practice Exam", "time_limit": 60}
        try:
            res_ex = supabase.table("exam_profiles").select("*").eq("id", exam_id).execute()
            if res_ex.data:
                exam_profile = res_ex.data[0]
        except Exception:
            pass

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
            "created_at": datetime.now(timezone.utc).isoformat()
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
