import uuid
from typing import Any, Dict, List, Optional
from database import supabase
from services.ollama_service import ollama_service
from services.summary_service import summary_service

# In-memory stores for unit tests / local fallback
_in_memory_questions: Dict[str, List[Dict[str, Any]]] = {}
_in_memory_attempts: List[Dict[str, Any]] = []
_reviewed_weak_areas: set = set() # (user_id, document_id, section_index, page_reference)

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
        section_stats: Dict[int, Dict[str, int]] = {}

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
                "document_id": document_id,
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

        weak_areas = await self.get_document_weak_areas(user_id, document_id)

        return {
            "document_id": document_id,
            "score": score_pct,
            "total_questions": total_questions,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "breakdown": breakdown,
            "weak_areas": weak_areas
        }

    async def get_document_weak_areas(self, user_id: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Aggregates incorrect attempts by section and page for a specific document.
        Prioritizes sections/pages with low confidence score (<=2) AND incorrect answers.
        """
        questions = await self.get_quiz_questions(document_id)
        q_map = {q["id"]: q for q in questions}

        # Gather user attempts
        user_attempts = [a for a in _in_memory_attempts if a.get("user_id") == user_id]
        if document_id:
            user_attempts = [a for a in user_attempts if a.get("document_id") == document_id or a.get("question_id") in q_map]

        try:
            res = supabase.table("quiz_attempts").select("*").eq("user_id", user_id).execute()
            if res.data:
                db_attempts = [a for a in res.data if a.get("question_id") in q_map]
                user_attempts.extend(db_attempts)
        except Exception:
            pass

        summary = await summary_service.get_latest_summary(document_id)
        sections = summary.get("sections", []) if summary else []

        aggregated: Dict[tuple, Dict[str, Any]] = {} # (section_index, page_reference) -> stats

        for att in user_attempts:
            q_id = att.get("question_id")
            q = q_map.get(q_id)
            if not q:
                continue

            sec_idx = q.get("section_index", 0)
            page_ref = q.get("page_reference", 1)
            key = (sec_idx, page_ref)

            if key in _reviewed_weak_areas or (user_id, document_id, sec_idx, page_ref) in _reviewed_weak_areas:
                continue

            if key not in aggregated:
                sec_title = f"Section {sec_idx+1}"
                if sections and sec_idx < len(sections):
                    sec_title = sections[sec_idx].get("title", sec_title)

                aggregated[key] = {
                    "document_id": document_id,
                    "section_index": sec_idx,
                    "section_title": sec_title,
                    "page_reference": page_ref,
                    "miss_count": 0,
                    "low_confidence_miss_count": 0,
                    "priority_score": 0.0,
                    "reviewed": False
                }

            if not att.get("is_correct"):
                aggregated[key]["miss_count"] += 1
                conf = att.get("confidence_score", 3)
                if conf <= 2:
                    aggregated[key]["low_confidence_miss_count"] += 1

        results = []
        for key, stats in aggregated.items():
            if stats["miss_count"] > 0:
                # Priority calculation: base miss_count + 2x multiplier for low confidence misses
                priority = stats["miss_count"] + (stats["low_confidence_miss_count"] * 2)
                stats["priority_score"] = float(priority)
                results.append(stats)

        results.sort(key=lambda x: (x["priority_score"], x["miss_count"]), reverse=True)
        return results

    async def mark_weak_area_reviewed(self, user_id: str, document_id: str, section_index: int, page_reference: int) -> bool:
        """Marks a specific weak area as reviewed."""
        _reviewed_weak_areas.add((user_id, document_id, section_index, page_reference))
        return True

    async def get_dashboard_weak_areas(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Aggregates weak areas across ALL documents for the user,
        prioritizing sections/pages with low confidence and high miss counts.
        """
        all_attempts = [a for a in _in_memory_attempts if a.get("user_id") == user_id]

        try:
            res = supabase.table("quiz_attempts").select("*").eq("user_id", user_id).execute()
            if res.data:
                all_attempts.extend(res.data)
        except Exception:
            pass

        # Collect all questions
        all_q_ids = list(set([a.get("question_id") for a in all_attempts if a.get("question_id")]))
        q_map = {}
        for doc_qs in _in_memory_questions.values():
            for q in doc_qs:
                q_map[q["id"]] = q

        try:
            if all_q_ids:
                res_q = supabase.table("quiz_questions").select("*").in_("id", all_q_ids).execute()
                if res_q.data:
                    for q in res_q.data:
                        q_map[q["id"]] = q
        except Exception:
            pass

        aggregated: Dict[tuple, Dict[str, Any]] = {} # (doc_id, sec_idx, page_ref) -> stats

        for att in all_attempts:
            q_id = att.get("question_id")
            q = q_map.get(q_id)
            if not q:
                continue

            doc_id = q.get("document_id") or att.get("document_id", "unknown_doc")
            sec_idx = q.get("section_index", 0)
            page_ref = q.get("page_reference", 1)
            key = (doc_id, sec_idx, page_ref)

            if (user_id, doc_id, sec_idx, page_ref) in _reviewed_weak_areas:
                continue

            if key not in aggregated:
                aggregated[key] = {
                    "document_id": doc_id,
                    "section_index": sec_idx,
                    "section_title": f"Section {sec_idx+1}",
                    "page_reference": page_ref,
                    "miss_count": 0,
                    "low_confidence_miss_count": 0,
                    "priority_score": 0.0,
                    "reviewed": False
                }

            if not att.get("is_correct"):
                aggregated[key]["miss_count"] += 1
                if att.get("confidence_score", 3) <= 2:
                    aggregated[key]["low_confidence_miss_count"] += 1

        results = []
        for key, stats in aggregated.items():
            if stats["miss_count"] > 0:
                priority = stats["miss_count"] + (stats["low_confidence_miss_count"] * 2)
                stats["priority_score"] = float(priority)
                results.append(stats)

        results.sort(key=lambda x: (x["priority_score"], x["miss_count"]), reverse=True)
        return results

    async def submit_attempt(self, user_id: str, question_id: str, selected_answer: int, confidence_score: int) -> Dict[str, Any]:
        """Single attempt submission compatibility helper."""
        res = await self.submit_quiz_batch(user_id, "", [{
            "question_id": question_id,
            "selected_answer": selected_answer,
            "confidence_score": confidence_score
        }])
        return res["breakdown"][0] if res["breakdown"] else {}

quiz_service = QuizService()
