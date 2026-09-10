from typing import Optional
from supabase import Client, create_client
from config import settings

_supabase_client: Optional[Client] = None

def get_supabase_client() -> Client:
    """Returns initialized Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    return _supabase_client

supabase: Client = get_supabase_client()
