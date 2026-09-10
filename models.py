from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field

class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    VIDEO_READY = "video_ready"
    LOW_CONFIDENCE = "low_confidence"

class SourceType(str, Enum):
    PDF = "pdf"
    IMAGE_NOTES = "image_notes"

# --- AUTH MODELS ---

class SignUpRequest(BaseModel):
    email: str
    password: str
    grade_level: Optional[str] = None
    exam_targets: Optional[List[str]] = []

class LoginRequest(BaseModel):
    email: str
    password: str

class UserMetadata(BaseModel):
    grade_level: Optional[str] = None
    exam_targets: Optional[List[str]] = []

class UserProfileResponse(BaseModel):
    id: str
    email: str
    grade_level: Optional[str] = None
    exam_targets: Optional[List[str]] = []

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserProfileResponse

# --- DOCUMENT MERGE MODELS ---

class DocumentMergeRequest(BaseModel):
    document_ids: List[str]
    merged_filename: Optional[str] = "Merged_Document.pdf"

# --- TIMED EXAM MODE MODELS ---

class TimedExamStartRequest(BaseModel):
    subject: str
    document_ids: List[str] = []
    num_questions: Optional[int] = Field(35, ge=30, le=50)

class TimedExamAnswerItem(BaseModel):
    question_id: str
    selected_answer: int
    confidence_score: Optional[int] = 3

class TimedExamSubmissionRequest(BaseModel):
    exam_id: Optional[str] = None
    answers: List[TimedExamAnswerItem] = []

class TimedExamStatusResponse(BaseModel):
    exam_id: str
    subject: str
    time_remaining_seconds: int
    current_question: int
    total_questions: int
    progress_percent: float
    paused: bool

# --- ADMIN EXAM MODELS ---

class ExamResearchRequest(BaseModel):
    exam_name: str

class ExamProfileUpdate(BaseModel):
    exam_name: Optional[str] = None
    sections: Optional[List[Any]] = None
    question_types: Optional[List[str]] = None
    time_limit: Optional[int] = None
    total_questions: Optional[int] = None
    scoring_rules: Optional[str] = None
    disclaimer_text: Optional[str] = None

class ExamProfileFull(BaseModel):
    id: str
    exam_name: str
    sections: List[Any] = []
    question_types: List[str] = []
    time_limit: Optional[int] = None
    total_questions: Optional[int] = None
    scoring_rules: Optional[str] = None
    disclaimer_text: str

    model_config = ConfigDict(from_attributes=True)

# --- EXAM PAPER GENERATOR MODELS ---

EXAM_PAPER_DISCLAIMER = "This is an unofficial practice paper and is not affiliated with or endorsed by the exam board."

class ExamPaperGenerateRequest(BaseModel):
    document_ids: List[str] = []
    num_questions: int = Field(25, ge=1, le=100)

class ExamPaperAnswerItem(BaseModel):
    question_id: str
    selected_answer: int
    confidence_score: Optional[int] = 3

class ExamPaperSubmission(BaseModel):
    answers: List[ExamPaperAnswerItem]

class ExamPaperResponse(BaseModel):
    id: str
    exam_id: str
    exam_name: str
    questions: List[Dict[str, Any]]
    time_limit: Optional[int] = 60
    total_questions: int
    disclaimer_text: str = EXAM_PAPER_DISCLAIMER

    model_config = ConfigDict(from_attributes=True)

class ExamPaperGradeResponse(BaseModel):
    paper_id: str
    exam_id: str
    score: float
    total_questions: int
    correct_count: int
    incorrect_count: int
    breakdown: List[Dict[str, Any]]
    disclaimer_text: str = EXAM_PAPER_DISCLAIMER

# --- STUDY PLAN BUILDER MODELS ---

class StudyPlanCreateAI(BaseModel):
    exam_date: datetime
    subject: str
    documents: Optional[List[str]] = []

class StudyPlanUpdateAI(BaseModel):
    exam_date: Optional[datetime] = None
    subject: Optional[str] = None
    documents: Optional[List[str]] = None

class StudyPlanTaskItem(BaseModel):
    id: str
    type: str  # review, quiz, video, read
    document_id: Optional[str] = None
    description: str
    completed: bool = False

class StudyPlanDaySchedule(BaseModel):
    day: int
    date: Optional[str] = None
    tasks: List[StudyPlanTaskItem] = []

class StudyPlanFull(BaseModel):
    id: str
    user_id: str
    exam_date: datetime
    subject: str
    schedule: List[Dict[str, Any]] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# --- DOMAIN MODELS ---

class DocumentBase(BaseModel):
    filename: str
    file_url: Optional[str] = None
    extracted_text: Optional[str] = None
    status: DocumentStatus = DocumentStatus.PROCESSING
    page_count: Optional[int] = 0
    source_type: Optional[SourceType] = SourceType.PDF
    original_images: Optional[List[str]] = []
    virtual_page_map: Optional[List[Dict[str, Any]]] = []

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: str
    user_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ImageNotesUploadResponse(BaseModel):
    id: str
    filename: str
    status: DocumentStatus
    source_type: SourceType = SourceType.IMAGE_NOTES
    page_count: int
    virtual_page_map: List[Dict[str, Any]]
    ocr_confidence: float
    message: Optional[str] = None

class SummaryDepth(str, Enum):
    QUICK = "quick"
    DEEP = "deep"

class SummaryCreate(BaseModel):
    document_id: str
    depth: SummaryDepth = SummaryDepth.QUICK

class SummaryResponse(BaseModel):
    id: str
    document_id: str
    sections: List[Dict[str, Any]]
    depth: SummaryDepth
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class QuizQuestionResponse(BaseModel):
    id: str
    document_id: str
    section_index: int
    question_text: str
    options: List[str]
    correct_answer_index: int
    page_reference: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)

class QuizAttemptCreate(BaseModel):
    question_id: str
    selected_answer: int
    confidence_score: int = Field(ge=1, le=5)

class QuizAttemptResponse(BaseModel):
    id: str
    user_id: str
    question_id: str
    selected_answer: int
    is_correct: bool
    confidence_score: int
    answered_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ReviewItemResponse(BaseModel):
    id: str
    user_id: str
    question_id: str
    last_reviewed: datetime
    ease_factor: float
    interval: int
    next_review_date: datetime

    model_config = ConfigDict(from_attributes=True)

class StudyPlanCreate(BaseModel):
    exam_date: datetime
    documents: Optional[List[str]] = []

class StudyPlanResponse(BaseModel):
    id: str
    user_id: str
    exam_date: datetime
    schedule: List[Dict[str, Any]]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ExamProfileResponse(BaseModel):
    id: str
    exam_name: str
    sections: List[Dict[str, Any]]
    question_types: List[str]
    time_limit: Optional[int] = None
    disclaimer_text: str

    model_config = ConfigDict(from_attributes=True)

class UserUsageResponse(BaseModel):
    id: str
    user_id: str
    month: int
    year: int
    video_generations_used: int
    summary_generations_used: int
    quiz_generations_used: Optional[int] = 0

    model_config = ConfigDict(from_attributes=True)

class UserUsageStatsResponse(BaseModel):
    user_id: str
    month: int
    year: int
    video_generations_used: int
    video_generations_limit: int = 10
    summary_generations_used: int
    summary_generations_limit: int = 50
    quiz_generations_used: int = 0
    quiz_generations_limit: Optional[str] = "unlimited"
    next_reset_date: str
    disclaimer: str = "This content was generated by AI locally. Usage limits apply monthly."

class AdminUsageAdjustRequest(BaseModel):
    video_generations_used: Optional[int] = None
    summary_generations_used: Optional[int] = None
    quiz_generations_used: Optional[int] = None
    video_generations_limit: Optional[int] = None
    summary_generations_limit: Optional[int] = None

class AdminUserStatusUpdate(BaseModel):
    status: str = Field(pattern="^(active|suspended)$")
    reason: Optional[str] = None

class AdminSystemLogItem(BaseModel):
    timestamp: datetime
    level: str
    component: str
    message: str
    details: Optional[Dict[str, Any]] = None

class SharedLinkCreate(BaseModel):
    document_id: Optional[str] = None
    expires_in_days: Optional[int] = 7

class SharedLinkResponse(BaseModel):
    id: str
    document_id: str
    share_id: str
    expires_at: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class PublicQuizQuestion(BaseModel):
    id: str
    section_index: int
    question_text: str
    options: List[str]

class PublicShareResponse(BaseModel):
    title: str
    summary_sections: List[Dict[str, Any]]
    quiz_questions: List[PublicQuizQuestion]
    expires_at: datetime
    disclaimer: str = "This public content was generated by AI locally. Please verify important information with primary source materials."

class PublicQuizSubmissionItem(BaseModel):
    question_id: str
    selected_answer: int

class PublicQuizSubmitRequest(BaseModel):
    submissions: List[PublicQuizSubmissionItem]

class PublicQuizSubmitResponse(BaseModel):
    score_percent: float
    correct_count: int
    total_questions: int
    breakdown: List[Dict[str, Any]]
    disclaimer: str = "This content was generated by AI locally. Scores are evaluated anonymously and not saved."

class AIContentDisclaimer(BaseModel):
    disclaimer: str = "This content was generated by AI locally. Please verify important information with primary source materials."
