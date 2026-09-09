import uuid
from typing import Any, Dict, List, Optional
from database import supabase
from services.ollama_service import ollama_service
from services.summary_service import summary_service

# In-memory stores for unit tests / local fallback
_in_memory_questions: Dict[str, List[Dict[str, Any]]] = {}
_in_memory_attempts: List[Dict[str, Any]] = []

class QuizService:
    """Service for generating quiz questions from summaries, evaluating attempts, and identifying weak study areas."""

    async def generate_quiz_from_summary(self, document_id: str, num_questions: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves document summary, queries Ollama for num_questions multiple choice questions
        with 4 options and correct answer index, and stores questions in database.
        """
        summary = await summary_service.get_latest_summary(document_id)
        summary_text = ""
        sections = []
        if summary and "sections" in summary:
            sections = summary["sections"]
            for idx, sec in enumerate(sections):
                title = sec.get("title", f"Section {idx+1}")
                bullets = sec.get("bullets", [])
                bullet_strs = [b.get("text", "") if isinstance(b, dict) else str(b) for b in bullets]
                summary_text += f"\nSection {idx+1}: {title}\n" + "\n".join(f"- {bs}" for bs in bullet_strs) + "\n"

        prompt = (
            f"Based on the following document summary, generate exactly {num_questions} multiple-choice quiz questions.\n"
            f"Requirements:\n"
            f"- Each question MUST have 4 option choices.\n"
            f"- 'correct_answer_index' must be an integer from 0 to 3.\n"
            f"- 'section_index' must indicate which section (0-indexed) the question relates to.\n"
            f"- Output MUST be valid JSON with key 'questions' containing list of objects:\n"
            f"  [{{\"section_index\": 0, \"question_text\": \"...\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"correct_answer_index\": 0, \"page_reference\": 1}}]\n\n"
            f"Summary Content:\n{summary_text if summary_text else 'General concepts and key facts.'}"
        )
        system = "You are an AI quiz generator. Create clear, challenging multiple choice questions for high school students."

        try:
            data = await ollama_service.generate_json(prompt, system_prompt=system)
            raw_questions = data.get("questions", [])
        except Exception:
            raw_questions = []

        if not raw_questions or len(raw_questions) < num_questions:
            # Fallback question generator if LLM response incomplete
            fallback_qs = []
            for i in range(num_questions):
                sec_idx = i % (len(sections) if sections else 1)
                sec_title = sections[sec_idx].get("title", "Main Topic") if sections else "Main Topic"
                fallback_qs.append({
                    "section_index": sec_idx,
                    "question_text": f"Which of the following best describes the key point of {sec_title} (Q{i+1})?",
                    "options": [
                        f"Correct insight regarding {sec_title}",
                        f"Incorrect alternative A for {sec_title}",
                        f"Incorrect alternative B for {sec_title}",
                        f"Incorrect alternative C for {sec_title}"
                    ],
                    "correct_answer_index": 0,
                    "page_reference": 1
                })
            raw_questions = fallback_qs

        saved_questions = []
        for q in raw_questions[:num_questions]:
            q_id = str(uuid.uuid4())
            record = {
                "id": q_id,
                "document_id": document_id,
                "section_index": int(q.get("section_index", 0)),
                "question_text": str(q.get("question_text", "Sample question")),
                "options": q.get("options", ["Option A", "Option B", "Option C", "Option D"]),
                "correct_answer_index": int(q.get("correct_answer_index", 0)),
                "page_reference": int(q.get("page_reference", 1))
            }

            try:
                res = supabase.table("quiz_questions").insert(record).execute()
                if res.data:
                    saved_questions.append(res.data[0])
                else:
                    saved_questions.append(record)
            except Exception:
                saved_questions.append(record)

        if document_id not in _in_memory_questions:
            _in_memory_questions[document_id] = []
        _in_memory_questions[document_id].extend(saved_questions)

        return saved_questions

    async def generate_quiz_questions(self, document_id: str, text: str, count: int = 5) -> List[Dict[str, Any]]:
        """Backwards compatibility wrapper."""
        return await self.generate_quiz_from_summary(document_id, num_questions=count)

    async def get_quiz_questions(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieves all quiz questions for a document."""
        try:
            res = supabase.table("quiz_questions").select("*").eq("document_id", document_id).execute()
            if res.data:
                return res.data
        except Exception:
            pass

        return _in_memory_questions.get(document_id, [])

    async def submit_quiz_batch(self, user_id: str, document_id: str, submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes batch quiz submission array of {question_id, selected_answer, confidence_score}.
        Returns score percentage, correct/incorrect count, question breakdown, and weak areas.
        """
        questions = await self.get_quiz_questions(document_id)
        question_map = {q["id"]: q for q in questions}

        total_questions = len(submissions)
        correct_count = 0
        incorrect_count = 0
        breakdown = []
        section_stats: Dict[int, Dict[str, int]] = {} # section_index -> {correct, total, low_confidence}

        for sub in submissions:
            q_id = sub.get("question_id")
            selected_ans = sub.get("selected_answer", 0)
            confidence = sub.get("confidence_score", 3)

            q = question_map.get(q_id, {})
            correct_ans = q.get("correct_answer_index", 0)
            sec_idx = q.get("section_index", 0)

            is_correct = (selected_ans == correct_ans)
            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1

            if sec_idx not in section_stats:
                section_stats[sec_idx] = {"correct": 0, "total": 0, "low_confidence": 0}
            section_stats[sec_idx]["total"] += 1
            if is_correct:
                section_stats[sec_idx]["correct"] += 1
            if confidence <= 2:
                section_stats[sec_idx]["low_confidence"] += 1

            attempt_record = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "question_id": q_id,
                "selected_answer": selected_ans,
                "is_correct": is_correct,
                "confidence_score": confidence,
            }
            _in_memory_attempts.append(attempt_record)

            try:
                supabase.table("quiz_attempts").insert(attempt_record).execute()
            except Exception:
                pass

            breakdown.append({
                "question_id": q_id,
                "question_text": q.get("question_text", ""),
                "selected_answer": selected_ans,
                "correct_answer": correct_ans,
                "is_correct": is_correct,
                "confidence_score": confidence,
                "section_index": sec_idx
            })

        score_pct = round((correct_count / total_questions) * 100, 2) if total_questions > 0 else 0.0

        # Identify weak areas (accuracy < 60% or high proportion of low confidence)
        weak_areas = []
        summary = await summary_service.get_latest_summary(document_id)
        sections = summary.get("sections", []) if summary else []

        for sec_idx, stats in section_stats.items():
            tot = stats["total"]
            accuracy = (stats["correct"] / tot) if tot > 0 else 0.0
            if accuracy < 0.60 or stats["low_confidence"] > 0:
                sec_title = f"Section {sec_idx+1}"
                if sections and sec_idx < len(sections):
                    sec_title = sections[sec_idx].get("title", sec_title)
                weak_areas.append({
                    "section_index": sec_idx,
                    "section_title": sec_title,
                    "accuracy": round(accuracy * 100, 1),
                    "low_confidence_count": stats["low_confidence"]
                })

        return {
            "document_id": document_id,
            "score": score_pct,
            "total_questions": total_questions,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "breakdown": breakdown,
            "weak_areas": weak_areas
        }

    async def submit_attempt(self, user_id: str, question_id: str, selected_answer: int, confidence_score: int) -> Dict[str, Any]:
        """Single attempt submission compatibility helper."""
        res = await self.submit_quiz_batch(user_id, "", [{
            "question_id": question_id,
            "selected_answer": selected_answer,
            "confidence_score": confidence_score
        }])
        return res["breakdown"][0] if res["breakdown"] else {}

quiz_service = QuizService()
