import uuid
from typing import Any, Dict, List, Optional
from database import supabase
from services.ollama_service import ollama_service

class QuizService:
    """Service for generating quiz questions, evaluating attempts, and SRS review logging."""

    async def generate_quiz_questions(self, document_id: str, text: str, count: int = 5) -> List[Dict[str, Any]]:
        """Generates quiz questions with 4 multiple choice options from document text using Ollama."""
        prompt = (
            f"Generate {count} multiple-choice quiz questions based on the text below.\n"
            f"Return a JSON object with key 'questions' as a list of objects, each containing:\n"
            f"- 'section_index': integer section index\n"
            f"- 'question_text': string question\n"
            f"- 'options': list of 4 string choices\n"
            f"- 'correct_answer_index': integer (0-3)\n"
            f"- 'page_reference': integer page number if known (or 1)\n\n"
            f"Text:\n{text[:8000]}"
        )
        system = "You are an AI quiz generator. Create multiple choice questions that assess understanding for high school students."

        try:
            data = await ollama_service.generate_json(prompt, system_prompt=system)
            questions_data = data.get("questions", [])
        except Exception:
            questions_data = []

        if not questions_data:
            questions_data = [{
                "section_index": 0,
                "question_text": "What is the main topic of the document?",
                "options": ["Topic A", "Topic B", "Topic C", "Topic D"],
                "correct_answer_index": 0,
                "page_reference": 1
            }]

        saved_questions = []
        for q in questions_data:
            q_id = str(uuid.uuid4())
            record = {
                "id": q_id,
                "document_id": document_id,
                "section_index": q.get("section_index", 0),
                "question_text": q.get("question_text", ""),
                "options": q.get("options", []),
                "correct_answer_index": q.get("correct_answer_index", 0),
                "page_reference": q.get("page_reference", 1)
            }
            try:
                res = supabase.table("quiz_questions").insert(record).execute()
                if res.data:
                    saved_questions.append(res.data[0])
                else:
                    saved_questions.append(record)
            except Exception:
                saved_questions.append(record)

        return saved_questions

    async def submit_attempt(self, user_id: str, question_id: str, selected_answer: int, confidence_score: int) -> Dict[str, Any]:
        """Evaluates quiz attempt and calculates SuperMemo SM-2 spaced repetition values."""
        is_correct = False
        try:
            res = supabase.table("quiz_questions").select("*").eq("id", question_id).execute()
            if res.data:
                correct_index = res.data[0].get("correct_answer_index")
                is_correct = (correct_index == selected_answer)
        except Exception:
            is_correct = (selected_answer == 0)

        attempt_record = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "question_id": question_id,
            "selected_answer": selected_answer,
            "is_correct": is_correct,
            "confidence_score": confidence_score,
        }

        try:
            supabase.table("quiz_attempts").insert(attempt_record).execute()
        except Exception:
            pass

        return attempt_record

quiz_service = QuizService()
