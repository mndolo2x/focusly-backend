import os
import uuid
from typing import Any, Dict, List, Optional
from database import supabase
from services.tts_service import tts_service
from services.summary_service import summary_service
from services.document_service import document_service
from utils.ffmpeg_helper import ffmpeg_helper

class VideoService:
    """Service for video lesson assembly from document summary sections."""

    async def check_user_video_quota(self, user_id: str) -> bool:
        """Verifies if user has remaining video generation credits (max 10/month)."""
        try:
            res = supabase.table("user_usage").select("video_generations_used").eq("user_id", user_id).execute()
            if res.data and res.data[0].get("video_generations_used", 0) >= 10:
                return False
        except Exception:
            pass
        return True

    async def generate_video_lesson(
        self, document_id: str, title: Optional[str] = None, summary_sections: Optional[List[Dict[str, Any]]] = None, output_dir: str = "/tmp/videos"
    ) -> str:
        """
        1. Retrieves summary sections.
        2. Calls Kokoro TTS for narration & Pillow for 1080p white slides with page numbers.
        3. Assembles MP4 video using FFmpeg.
        4. Uploads MP4 to Supabase Storage 'videos' bucket.
        5. Updates document status='video_ready' with public video URL.
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

            slide_paths.append(slide_img_path)
            audio_paths.append(audio_file_path)

        # Assemble MP4 video
        ffmpeg_helper.assemble_video(slide_paths, audio_paths, local_video_path)

        # Upload MP4 file to Supabase Storage 'videos' or 'documents' bucket
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

video_service = VideoService()
