import json
from typing import Any, Dict, List, Optional
import httpx
from config import settings

class OllamaService:
    """Service wrapper for communicating with local Ollama instance."""

    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, model: str = settings.OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate_completion(self, prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False) -> str:
        """Generates text completion using Ollama LLM."""
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
            except Exception as e:
                # Handle connection error gracefully / fallback for offline mock if server not running
                raise RuntimeError(f"Ollama generation failed: {str(e)}")

    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Any:
        """Generates JSON object using Ollama LLM."""
        res_text = await self.generate_completion(prompt, system_prompt=system_prompt, json_mode=True)
        try:
            return json.loads(res_text)
        except json.JSONDecodeError:
            return {"raw": res_text}

ollama_service = OllamaService()
