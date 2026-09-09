# Focusly Backend - AGENTS.md

## Project Overview
Focusly is a learning platform that converts PDF uploads into AI-generated summaries, quizzes, and narrated video lessons. Users are typically 14-18 years old, so data privacy and compliance are critical.

## Tech Stack
- **Framework**: Python FastAPI
- **Database**: Supabase (PostgreSQL)
- **Auth**: Supabase Auth
- **Background Tasks**: Celery + Redis
- **Local AI**: Ollama + Llama 3.1 8B (primary model)
- **Text-to-Speech**: Kokoro TTS (open-source, runs locally)
- **Video Assembly**: FFmpeg + Pillow
- **OCR**: Tesseract (for scanned PDFs)
- **Vector Search**: FAISS (for RAG/source tracking)
- **Storage**: Supabase Storage

## Core Principle: 100% Local AI
- NO external API calls to OpenAI, Claude, or any cloud AI service
- All AI processing uses Ollama running locally
- All TTS uses Kokoro running locally
- All video assembly uses FFmpeg locally

## Coding Standards
- Use Pydantic models for all request/response validation
- All endpoints return proper JSON with consistent formatting
- Async/await for all I/O operations
- Environment variables via .env (never hardcode secrets)
- Type hints on all functions
- Docstrings for all public functions

## Database Tables
- `documents`: id, user_id, filename, file_url, extracted_text, status (processing/completed/failed/video_ready), page_count, created_at
- `summaries`: id, document_id, sections (JSONB), depth (quick/deep), created_at
- `quiz_questions`: id, document_id, section_index, question_text, options (JSONB), correct_answer_index, page_reference
- `quiz_attempts`: id, user_id, question_id, selected_answer, is_correct, confidence_score (1-5), answered_at
- `review_items`: id, user_id, question_id, last_reviewed, ease_factor, interval, next_review_date
- `study_plans`: id, user_id, exam_date, schedule (JSONB), created_at
- `exam_profiles`: id, exam_name, sections (JSONB), question_types (JSONB), time_limit, disclaimer_text
- `user_usage`: id, user_id, month, year, video_generations_used, summary_generations_used
- `shared_links`: id, document_id, share_id, expires_at, created_at

## Security
- All endpoints except public shares require authentication
- Row Level Security in Supabase: users access only their own data
- Usage caps: 10 video generations/month, 50 summaries/month per user
- All AI content includes disclaimers
