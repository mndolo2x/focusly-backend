import base64
import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
import httpx
from PIL import Image

from config import settings

logger = logging.getLogger("focusly.gemini")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
    logger.addHandler(ch)

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class GeminiService:
    """Centralized AI service wrapper for Google Gemini API integration."""

    def __init__(self):
        self.api_key = getattr(settings, "GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
        self.text_model_name = getattr(settings, "GEMINI_TEXT_MODEL", os.getenv("GEMINI_TEXT_MODEL", "gemini-1.5-flash"))
        self.vision_model_name = getattr(settings, "GEMINI_VISION_MODEL", os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-flash"))
        self.embedding_model_name = getattr(settings, "GEMINI_EMBEDDING_MODEL", os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004"))

        if genai is not None and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to configure Google Generative AI client: {e}")

    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        """
        Generates text or JSON response from Google Gemini API with retry logic and error logging.
        """
        target_model = model or self.text_model_name
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System Instruction: {system_prompt}\n\nUser Request:\n{prompt}"

        logger.info(f"Gemini Request -> Model: {target_model}, Prompt len: {len(prompt)}, JSON Mode: {json_mode}")

        for attempt in range(1, max_retries + 1):
            try:
                if genai is not None and self.api_key:
                    gen_config = {}
                    if json_mode:
                        gen_config["response_mime_type"] = "application/json"

                    model_instance = genai.GenerativeModel(target_model, generation_config=gen_config if gen_config else None)
                    response = await model_instance.generate_content_async(full_prompt)
                    res_text = response.text if response and response.text else ""
                    logger.info(f"Gemini Response Success -> Model: {target_model}, Response len: {len(res_text)}")
                    return res_text
                else:
                    # Fallback HTTP REST API call if google.generativeai SDK is not initialized
                    if not self.api_key:
                        raise RuntimeError("GEMINI_API_KEY is not configured.")

                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"
                    payload = {
                        "contents": [{"parts": [{"text": full_prompt}]}]
                    }
                    if json_mode:
                        payload["generationConfig"] = {"responseMimeType": "application/json"}

                    async with httpx.AsyncClient(timeout=120.0) as client:
                        resp = await client.post(url, json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            res_text = "".join(p.get("text", "") for p in parts)
                            return res_text
                        raise RuntimeError(f"Unexpected Gemini REST response structure: {data}")

            except Exception as e:
                logger.warning(f"Gemini Attempt {attempt}/{max_retries} failed for model {target_model}: {str(e)}")
                if attempt == max_retries:
                    logger.error(f"Gemini max retries reached ({max_retries}). Request failed: {str(e)}")
                    raise RuntimeError(f"Focusly couldn't process this material right now. Please try again. ({str(e)})")
                time.sleep(retry_delay * attempt)

        raise RuntimeError("Gemini generate_text failed unexpectedly")

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Any:
        """
        Generates and parses structured JSON from Gemini API.
        """
        res_text = await self.generate_text(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            json_mode=True
        )
        try:
            return json.loads(res_text)
        except json.JSONDecodeError:
            # Fallback regex JSON extraction
            import re
            match = re.search(r'\{.*\}|\[.*\]', res_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return {"raw_response": res_text}

    async def generate_with_image(
        self,
        prompt: str,
        image_input: Any,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Multimodal image analysis using Gemini Vision model.
        Accepts PIL Image, base64 string, or bytes.
        """
        target_model = model or self.vision_model_name

        pil_image = None
        if isinstance(image_input, Image.Image):
            pil_image = image_input
        elif isinstance(image_input, bytes):
            pil_image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, str):
            if os.path.exists(image_input):
                pil_image = Image.open(image_input)
            else:
                # Assume base64 string
                img_bytes = base64.b64decode(image_input)
                pil_image = Image.open(io.BytesIO(img_bytes))

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System Instruction: {system_prompt}\n\nUser Prompt:\n{prompt}"

        logger.info(f"Gemini Vision Request -> Model: {target_model}, Prompt len: {len(prompt)}")

        for attempt in range(1, max_retries + 1):
            try:
                if genai is not None and self.api_key and pil_image:
                    model_instance = genai.GenerativeModel(target_model)
                    response = await model_instance.generate_content_async([full_prompt, pil_image])
                    res_text = response.text if response and response.text else ""
                    return {"response": res_text, "text": res_text}
                else:
                    # HTTP REST Fallback
                    if not self.api_key:
                        raise RuntimeError("GEMINI_API_KEY is not configured.")

                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="JPEG")
                    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"
                    payload = {
                        "contents": [
                            {
                                "parts": [
                                    {"text": full_prompt},
                                    {
                                        "inlineData": {
                                            "mimeType": "image/jpeg",
                                            "data": base64_str
                                        }
                                    }
                                ]
                            }
                        ]
                    }

                    async with httpx.AsyncClient(timeout=120.0) as client:
                        resp = await client.post(url, json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            res_text = "".join(p.get("text", "") for p in parts)
                            return {"response": res_text, "text": res_text}
                        raise RuntimeError(f"Unexpected Gemini Vision response: {data}")

            except Exception as e:
                logger.warning(f"Gemini Vision Attempt {attempt}/{max_retries} failed: {str(e)}")
                if attempt == max_retries:
                    raise RuntimeError(f"Focusly couldn't process this image notes right now. ({str(e)})")
                time.sleep(1.0)

        raise RuntimeError("Gemini generate_with_image failed unexpectedly")

    async def embed_text(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """
        Generates vector embeddings using Gemini embedding models.
        """
        target_model = model or self.embedding_model_name
        logger.info(f"Gemini Embed Request -> Model: {target_model}, Text len: {len(text)}")

        try:
            if genai is not None and self.api_key:
                res = genai.embed_content(
                    model=f"models/{target_model}",
                    content=text,
                    task_type="retrieval_document"
                )
                embedding = res.get("embedding", [])
                return embedding
            else:
                if not self.api_key:
                    raise RuntimeError("GEMINI_API_KEY is not configured.")

                url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:embedContent?key={self.api_key}"
                payload = {
                    "model": f"models/{target_model}",
                    "content": {"parts": [{"text": text}]}
                }

                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    embedding = data.get("embedding", {}).get("values", [])
                    return embedding
        except Exception as e:
            logger.warning(f"Gemini Embed Failed: {str(e)}")
            # Fallback deterministic vector if embedding service is offline
            import hashlib
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
            import random
            rng = random.Random(seed)
            return [rng.uniform(-1.0, 1.0) for _ in range(768)]

    async def ask_focusly(
        self,
        question: str,
        context_text: Optional[str] = None,
        student_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Interactive Q&A chatbot ("Ask Focusly") grounded in student documents and history.
        """
        system = (
            "You are Focusly AI, a friendly, patient, and highly articulate tutor for high school and college students.\n"
            "Your goal is to answer student questions clearly, provide examples, explain why wrong answers occur, "
            "and ground explanations in the provided document material whenever available."
        )

        prompt_parts = []
        if context_text:
            prompt_parts.append(f"Document Study Material Context:\n{context_text[:10000]}\n")
        if student_history:
            prompt_parts.append(f"Student Recent Quiz/Weak Area Context:\n{json.dumps(student_history[:5])}\n")

        prompt_parts.append(f"Student Question: {question}")
        full_prompt = "\n\n".join(prompt_parts)

        answer_text = await self.generate_text(full_prompt, system_prompt=system)

        return {
            "question": question,
            "answer": answer_text,
            "grounded_in_context": bool(context_text),
            "timestamp": time.time()
        }

    async def check_health(self) -> Dict[str, Any]:
        """Checks Gemini API configuration and connectivity."""
        if not self.api_key:
            return {
                "status": "offline",
                "provider": "google_gemini",
                "api_key_configured": False,
                "text_model": self.text_model_name,
                "vision_model": self.vision_model_name,
                "embedding_model": self.embedding_model_name,
                "error": "GEMINI_API_KEY environment variable is not configured."
            }

        try:
            # Test text generation
            test_res = await self.generate_text("Hello Focusly", max_retries=1)
            return {
                "status": "online",
                "provider": "google_gemini",
                "api_key_configured": True,
                "text_model": self.text_model_name,
                "vision_model": self.vision_model_name,
                "embedding_model": self.embedding_model_name,
                "primary_model_ready": bool(test_res)
            }
        except Exception as e:
            return {
                "status": "degraded",
                "provider": "google_gemini",
                "api_key_configured": True,
                "text_model": self.text_model_name,
                "vision_model": self.vision_model_name,
                "embedding_model": self.embedding_model_name,
                "primary_model_ready": False,
                "error": str(e)
            }


gemini_service = GeminiService()
