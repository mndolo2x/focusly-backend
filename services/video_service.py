import os
import uuid
from typing import Any, Dict, List, Optional
from database import supabase
from services.tts_service import tts_service
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

    async def generate_video_lesson(self, document_id: str, title: str, summary_sections: List[Dict[str, Any]], output_dir: str = "/tmp/videos") -> str:
        """
        Generates narration for each section, renders slide images, and stitches them into a final MP4 video lesson.
        """
        os.makedirs(output_dir, exist_ok=True)
        video_filename = f"lesson_{document_id}_{uuid.uuid4().hex[:8]}.mp4"
        final_video_path = os.path.join(output_dir, video_filename)

        slide_paths = []
        audio_paths = []

        for idx, sec in enumerate(summary_sections):
            sec_title = sec.get("title", f"Section {idx+1}")
            sec_text = sec.get("explanation", "") or " ".join(sec.get("key_points", []))

            slide_img_path = os.path.join(output_dir, f"slide_{idx}.png")
            ffmpeg_helper.create_slide_image(sec_title, sec_text, slide_img_path)

            audio_file_path = os.path.join(output_dir, f"narration_{idx}.wav")
            await tts_service.text_to_speech(f"{sec_title}. {sec_text}", audio_file_path)

            slide_paths.append(slide_img_path)
            audio_paths.append(audio_file_path)

        # Assemble video with FFmpeg
        ffmpeg_helper.assemble_video(slide_paths, audio_paths, final_video_path)

        # Update document status to video_ready
        try:
            supabase.table("documents").update({"status": "video_ready", "file_url": final_video_path}).eq("id", document_id).execute()
        except Exception:
            pass

        return final_video_path

video_service = VideoService()
