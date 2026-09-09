import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import httpx
from config import settings

logger = logging.getLogger("focusly.ollama")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
    logger.addHandler(ch)

class OllamaService:
    """Service wrapper for communicating with local Ollama server."""

    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, default_model: str = settings.OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    async def generate(
        self,
        prompt: str,
        model: str = "llama3.1:8b",
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        """
        Generates text or JSON response from local Ollama model with retry logic and logging.
        """
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if json_mode:
            payload["format"] = "json"

        logger.info(f"Ollama Generate Request -> Model: {model}, URL: {url}, Prompt len: {len(prompt)}")

        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    res_text = data.get("response", "")
                    logger.info(f"Ollama Generate Response Success -> Model: {model}, Response len: {len(res_text)}")
                    return res_text
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.warning(
                    f"Ollama Generate Attempt {attempt}/{max_retries} failed for model {model}: {str(e)}"
                )
                if attempt == max_retries:
                    logger.error(f"Ollama Generate max retries reached ({max_retries}). Request failed: {str(e)}")
                    raise RuntimeError(f"Ollama connection error: {str(e)}")
                await asyncio.sleep(retry_delay * attempt)

        raise RuntimeError("Ollama generate failed unexpectedly")

    async def embed(
        self,
        text: str,
        model: str = "nomic-embed-text",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> List[float]:
        """
        Generates vector embedding for input text using local Ollama embedding model.
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model,
            "prompt": text,
        }

        logger.info(f"Ollama Embed Request -> Model: {model}, URL: {url}, Text len: {len(text)}")

        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    embedding = data.get("embedding", [])
                    logger.info(f"Ollama Embed Response Success -> Model: {model}, Vector dim: {len(embedding)}")
                    return embedding
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.warning(
                    f"Ollama Embed Attempt {attempt}/{max_retries} failed for model {model}: {str(e)}"
                )
                if attempt == max_retries:
                    logger.error(f"Ollama Embed max retries reached ({max_retries}). Request failed: {str(e)}")
                    raise RuntimeError(f"Ollama embedding connection error: {str(e)}")
                await asyncio.sleep(retry_delay * attempt)

        raise RuntimeError("Ollama embed failed unexpectedly")

    async def check_health(self) -> Dict[str, Any]:
        """Checks connection to Ollama server and lists loaded/installed models."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name") for m in data.get("models", [])]
                    primary_ready = any("llama3.1:8b" in m or "llama3.1" in m for m in models)
                    fallback_ready = any("phi3" in m for m in models)
                    embed_ready = any("nomic-embed-text" in m for m in models)
                    return {
                        "status": "online",
                        "base_url": self.base_url,
                        "available_models": models,
                        "primary_model_ready": primary_ready,
                        "fallback_model_ready": fallback_ready,
                        "embedding_model_ready": embed_ready,
                    }
        except Exception as e:
            logger.warning(f"Ollama Health Check Failed: {str(e)}")

        return {
            "status": "offline",
            "base_url": self.base_url,
            "available_models": [],
            "primary_model_ready": False,
            "fallback_model_ready": False,
            "embedding_model_ready": False,
            "error": "Ollama service unavailable",
        }

    # Backwards compatibility methods
    async def generate_completion(self, prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False) -> str:
        return await self.generate(prompt, model=self.default_model, system_prompt=system_prompt, json_mode=json_mode)

    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Any:
        res_text = await self.generate(prompt, model=self.default_model, system_prompt=system_prompt, json_mode=True)
        try:
            return json.loads(res_text)
        except json.JSONDecodeError:
            return {"raw": res_text}

ollama_service = OllamaService()
