from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from config import settings
from database import get_supabase_client
from models import SignUpRequest, LoginRequest, UserProfileResponse, TokenResponse

security = HTTPBearer(auto_error=False)
router = APIRouter(prefix="/api/auth", tags=["auth"])

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Validates JWT token from Authorization header or Supabase auth endpoint.
    Returns user payload dict containing user_id and metadata.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    try:
        if settings.SUPABASE_JWT_SECRET and settings.SUPABASE_JWT_SECRET != "your-supabase-jwt-secret":
            payload = jwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                options={"verify_aud": False}
            )
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
            user_meta = payload.get("user_metadata", {})
            return {
                "user_id": user_id,
                "email": payload.get("email"),
                "grade_level": user_meta.get("grade_level"),
                "exam_targets": user_meta.get("exam_targets", []),
                "token": token
            }
        else:
            supabase_client = get_supabase_client()
            response = supabase_client.auth.get_user(token)
            if not response or not response.user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user session")
            user_meta = getattr(response.user, "user_metadata", {}) or {}
            return {
                "user_id": response.user.id,
                "email": response.user.email,
                "grade_level": user_meta.get("grade_level"),
                "exam_targets": user_meta.get("exam_targets", []),
                "token": token
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/signup", response_model=UserProfileResponse)
async def signup(payload: SignUpRequest):
    """Creates a new user with email, password, and metadata in Supabase Auth."""
    supabase_client = get_supabase_client()
    meta = {
        "grade_level": payload.grade_level,
        "exam_targets": payload.exam_targets or []
    }
    try:
        res = supabase_client.auth.sign_up({
            "email": payload.email,
            "password": payload.password,
            "options": {"data": meta}
        })
        if not res.user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Signup failed")

        user_meta = getattr(res.user, "user_metadata", {}) or meta
        return UserProfileResponse(
            id=res.user.id,
            email=res.user.email or payload.email,
            grade_level=user_meta.get("grade_level"),
            exam_targets=user_meta.get("exam_targets", [])
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Signup error: {str(e)}")

@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest):
    """Authenticates user with email/password and returns Supabase session tokens."""
    supabase_client = get_supabase_client()
    try:
        res = supabase_client.auth.sign_in_with_password({
            "email": payload.email,
            "password": payload.password
        })
        if not res.session or not res.user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        user_meta = getattr(res.user, "user_metadata", {}) or {}
        profile = UserProfileResponse(
            id=res.user.id,
            email=res.user.email or payload.email,
            grade_level=user_meta.get("grade_level"),
            exam_targets=user_meta.get("exam_targets", [])
        )
        return TokenResponse(
            access_token=res.session.access_token,
            refresh_token=res.session.refresh_token,
            token_type="bearer",
            user=profile
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Login failed: {str(e)}")

@router.get("/me", response_model=UserProfileResponse)
async def get_me(user: Dict[str, Any] = Depends(get_current_user)):
    """Returns current user profile from validated JWT session."""
    return UserProfileResponse(
        id=user["user_id"],
        email=user.get("email", ""),
        grade_level=user.get("grade_level"),
        exam_targets=user.get("exam_targets", [])
    )

@router.post("/logout")
async def logout(user: Dict[str, Any] = Depends(get_current_user)):
    """Invalidates active user session."""
    try:
        supabase_client = get_supabase_client()
        supabase_client.auth.sign_out()
    except Exception:
        pass
    return {"message": "Successfully logged out"}
