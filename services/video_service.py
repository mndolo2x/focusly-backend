import os
import uuid
from typing import Any, Dict, List, Optional
from database import supabase
from services.tts_service import tts_service
from services.summary_service import summary_service
from services.document_service import document_service
from utils.ffmpeg_helper import ffmpeg_helper

# In-memory store for transcripts and captions (fallback / local testing)
_in_memory_transcripts: Dict[str, List[Dict[str, Any]]] = {}

class VideoService:
    """Service for video lesson assembly, section transcript tracking with timestamps, and WebVTT captions."""

    async def check_user_video_quota(self, user_id: str) -> bool:
        """Verifies if user has remaining video generation credits (max 10/month)."""
        try:
            res = supabase.table("user_usage").select("video_generations_used").eq("user_id", user_id).execute()
            if res.data and res.data[0].get("video_generations_used", 0) >= 10:
                return False
        except Exception:
            pass
        return True

    async def increment_video_usage(self, user_id: str) -> None:
        """Increments video_generations_used counter for user in user_usage table."""
        try:
            res = supabase.table("user_usage").select("video_generations_used").eq("user_id", user_id).execute()
            if res.data:
                curr = res.data[0].get("video_generations_used", 0)
                supabase.table("user_usage").update({"video_generations_used": curr + 1}).eq("user_id", user_id).execute()
        except Exception:
            pass

    def _format_timestamp_vtt(self, seconds: float) -> str:
        """Formats seconds into WebVTT timestamp format HH:MM:SS.mmm."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"

    async def generate_lesson_script(
        self, document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Generates structured lesson scripts (narration, visual cues, captions, quiz checkpoints) using Gemini.
        Separates script generation from media assembly.
        """
        summary = await summary_service.get_latest_summary(document_id)
        sections = summary.get("sections", []) if summary else []

        from services.gemini_service import gemini_service
        import json

        prompt = (
            f"Generate a structured video lesson script for the following study material.\n"
            f"Output JSON Format:\n"
            f'{{"script_sections": [{{"section_title": "...", "narration_text": "...", "visual_cue": "...", "quiz_checkpoint": "...", "page_number": 1}}]}}\n\n'
            f"Summary Content:\n{json.dumps(sections)}"
        )
        system = "You are an expert educational video scriptwriter producing clean, engaging lesson scripts."

        try:
            data = await gemini_service.generate_json(prompt, system_prompt=system)
            return data.get("script_sections", [])
        except Exception:
            return [{
                "section_title": "Introduction",
                "narration_text": "Welcome to this AI-narrated study lesson.",
                "visual_cue": "Display title slide and overview bullets.",
                "quiz_checkpoint": "What is the main topic of this lesson?",
                "page_number": 1
            }]

    async def generate_video_lesson(
        self, document_id: str, title: Optional[str] = None, summary_sections: Optional[List[Dict[str, Any]]] = None, output_dir: str = "/tmp/videos"
    ) -> str:
        """
        1. Retrieves summary sections.
        2. Calls Kokoro TTS for narration & Pillow for 1080p white slides with page numbers.
        3. Measures exact audio duration per section and builds transcript with timestamps (start_time, end_time).
        4. Assembles MP4 video using FFmpeg.
        5. Uploads MP4 to Supabase Storage 'videos' bucket.
        6. Updates document status='video_ready' with public video URL.
        """
        os.makedirs(output_dir, exist_ok=True)
        video_filename = f"video_{document_id}_{uuid.uuid4().hex[:8]}.mp4"
        local_video_path = os.path.join(output_dir, video_filename)

        if not summary_sections:
            summary = await summary_service.get_latest_summary(document_id)
            if summary and "sections" in summary:
                summary_sections = summary["sections"]

        if not summary_sections:
            summary_sections = [
                {
                    "title": title or "Lesson Summary",
                    "bullets": [{"text": "Key concept overview and summary points.", "page_number": 1}]
                }
            ]

        slide_paths = []
        audio_paths = []
        transcript_sections = []
        current_time = 0.0

        for idx, sec in enumerate(summary_sections):
            sec_title = sec.get("title", f"Section {idx+1}")
            bullets = sec.get("bullets", [])

            bullet_texts = []
            page_num = 1
            for b in bullets:
                if isinstance(b, dict):
                    bullet_texts.append(b.get("text", ""))
                    if "page_number" in b:
                        page_num = int(b["page_number"])
                elif isinstance(b, str):
                    bullet_texts.append(b)

            if not bullet_texts:
                bullet_texts = [sec.get("explanation", "Key concepts and details.")]

            narration_text = f"{sec_title}. " + " ".join(bullet_texts)

            slide_img_path = os.path.join(output_dir, f"slide_{document_id}_{idx}.png")
            ffmpeg_helper.create_slide_image(sec_title, bullet_texts, page_number=page_num, output_image_path=slide_img_path)

            audio_file_path = os.path.join(output_dir, f"narration_{document_id}_{idx}.mp3")
            await tts_service.generate_audio(narration_text, audio_file_path)

            # Measure precise audio duration for transcript timestamping
            duration = ffmpeg_helper.get_audio_duration(audio_file_path)
            end_time = current_time + duration

            transcript_sections.append({
                "section_index": idx,
                "title": sec_title,
                "text": narration_text,
                "start_time": round(current_time, 2),
                "end_time": round(end_time, 2),
                "duration": round(duration, 2),
                "page_number": page_num,
                "bullets": bullet_texts
            })

            current_time = end_time

            slide_paths.append(slide_img_path)
            audio_paths.append(audio_file_path)

        _in_memory_transcripts[document_id] = transcript_sections

        # Assemble MP4 video
        ffmpeg_helper.assemble_video(slide_paths, audio_paths, local_video_path)

        # Upload MP4 file to Supabase Storage
        video_url = local_video_path
        try:
            with open(local_video_path, "rb") as f:
                file_bytes = f.read()
            storage_path = f"videos/{document_id}/{video_filename}"
            supabase.storage.from_("documents").upload(
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": "video/mp4"}
            )
            public_url = supabase.storage.from_("documents").get_public_url(storage_path)
            if public_url:
                video_url = public_url
        except Exception:
            pass

        # Update document record status='video_ready' in database and in-memory store
        await document_service.update_document(document_id, {"status": "video_ready", "file_url": video_url})

        return video_url

    async def get_video_status(self, document_id: str) -> str:
        """Retrieves video generation status for document."""
        doc = await document_service.get_document(document_id, user_id="")
        if doc:
            return doc.get("status", "processing")
        try:
            res = supabase.table("documents").select("status").eq("id", document_id).execute()
            if res.data:
                return res.data[0].get("status", "processing")
        except Exception:
            pass
        return "ready"

    async def get_video_url(self, document_id: str) -> Optional[str]:
        """Retrieves public video URL for document."""
        doc = await document_service.get_document(document_id, user_id="")
        if doc and doc.get("status") == "video_ready":
            return doc.get("file_url")

        try:
            res = supabase.table("documents").select("file_url, status").eq("id", document_id).execute()
            if res.data:
                row = res.data[0]
                if row.get("status") == "video_ready":
                    return row.get("file_url")
        except Exception:
            pass
        return None

    async def get_document_transcript(self, document_id: str) -> List[Dict[str, Any]]:
        """Returns section-by-section transcript with timestamps and page tracking for document video."""
        if document_id in _in_memory_transcripts:
            return _in_memory_transcripts[document_id]

        # Generate fallback transcript if not previously stored
        summary = await summary_service.get_latest_summary(document_id)
        sections = summary.get("sections", []) if summary else []

        fallback_transcript = []
        curr_time = 0.0
        for idx, sec in enumerate(sections):
            sec_title = sec.get("title", f"Section {idx+1}")
            bullets = sec.get("bullets", [])
            bullet_texts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in bullets]
            page_num = bullets[0].get("page_number", 1) if bullets and isinstance(bullets[0], dict) else 1
            text = f"{sec_title}. " + " ".join(bullet_texts)
            duration = max(5.0, len(text.split()) * 0.4)
            end_time = curr_time + duration

            fallback_transcript.append({
                "section_index": idx,
                "title": sec_title,
                "text": text,
                "start_time": round(curr_time, 2),
                "end_time": round(end_time, 2),
                "duration": round(duration, 2),
                "page_number": page_num,
                "bullets": bullet_texts
            })
            curr_time = end_time

        if not fallback_transcript:
            fallback_transcript = [{
                "section_index": 0,
                "title": "Lesson Summary",
                "text": "Key concept overview.",
                "start_time": 0.0,
                "end_time": 5.0,
                "duration": 5.0,
                "page_number": 1,
                "bullets": ["Key concept overview."]
            }]

        _in_memory_transcripts[document_id] = fallback_transcript
        return fallback_transcript

    async def generate_webvtt_captions(self, document_id: str) -> str:
        """Formats section transcript into standard WebVTT caption string."""
        transcript = await self.get_document_transcript(document_id)

        vtt_lines = ["WEBVTT\n"]
        for idx, sec in enumerate(transcript):
            start_vtt = self._format_timestamp_vtt(sec.get("start_time", 0.0))
            end_vtt = self._format_timestamp_vtt(sec.get("end_time", 5.0))
            title = sec.get("title", "")
            text = sec.get("text", "")

            vtt_lines.append(f"{idx+1}")
            vtt_lines.append(f"{start_vtt} --> {end_vtt}")
            vtt_lines.append(f"[{title}] {text}\n")

        return "\n".join(vtt_lines)

video_service = VideoService()
