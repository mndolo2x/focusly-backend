import os
from json import JSONDecodeError, loads
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "https://your-supabase-project.supabase.co")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "your-supabase-anon-key")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    SUPABASE_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "your-supabase-jwt-secret")

    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    KOKORO_VOICE: str = os.getenv("KOKORO_VOICE", "af_heart")

    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    PORT: int = int(os.getenv("PORT", "8000"))

    @property
    def cors_origins(self) -> list[str]:
        raw = os.getenv("CORS_ORIGINS", '["*"]')
        try:
            parsed = loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (JSONDecodeError, TypeError):
            pass
        return [raw] if raw else ["*"]

    MAX_SUMMARY_GENERATIONS_PER_MONTH: int = int(os.getenv("MAX_SUMMARY_GENERATIONS_PER_MONTH", "50"))
    MAX_VIDEO_GENERATIONS_PER_MONTH: int = int(os.getenv("MAX_VIDEO_GENERATIONS_PER_MONTH", "10"))

    model_config = ConfigDict(env_file=".env", extra="ignore")

settings = Settings()
