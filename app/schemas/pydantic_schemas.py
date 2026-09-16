from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, EmailStr, Field

# User & Auth Schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class TokenData(BaseModel):
    user_id: Optional[str] = None


# Document Schemas
class DocumentChunkResponse(BaseModel):
    id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    char_count: int
    token_count: int
    vector_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    page_count: int
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class DocumentDetailResponse(DocumentResponse):
    chunks: List[DocumentChunkResponse] = []


# Source Citation Schema
class SourceCitation(BaseModel):
    document_id: str
    document_name: str
    page_number: int
    chunk_id: str
    similarity_score: float
    snippet: str


# Chat Schemas
class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    sender: str
    content: str
    sources: List[SourceCitation] = []
    created_at: datetime

    class Config:
        from_attributes = True

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True

class ChatSessionDetailResponse(ChatSessionResponse):
    messages: List[ChatMessageResponse] = []

class ChatQueryRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    document_ids: Optional[List[str]] = None
    top_k: Optional[int] = 5

class ChatQueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    sources: List[SourceCitation]
    response_time_ms: float


# Document Comparison Schemas
class DocumentCompareRequest(BaseModel):
    document_a_id: str
    document_b_id: str
    prompt: Optional[str] = "What are the main similarities, differences, and key changes between these two documents?"

class CompareItem(BaseModel):
    topic: str
    doc_a_findings: str
    doc_b_findings: str
    analysis: str

class DocumentCompareResponse(BaseModel):
    document_a_name: str
    document_b_name: str
    summary_comparison: str
    items: List[CompareItem] = []
    sources_a: List[SourceCitation] = []
    sources_b: List[SourceCitation] = []


# Semantic Search Schemas
class SemanticSearchRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: Optional[int] = 10

class SemanticSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str
    similarity_score: float

class SemanticSearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SemanticSearchResult]


# Evaluation & Dashboard Stats
class DashboardStatsResponse(BaseModel):
    total_documents: int
    indexed_documents: int
    processing_documents: int
    failed_documents: int
    total_chunks: int
    total_queries: int
    total_sessions: int
    average_response_time_ms: float

class EvaluationRunRequest(BaseModel):
    top_k: int = 5

class EvaluationRunResponse(BaseModel):
    total_questions: int
    retrieval_hit_rate: float
    retrieval_mrr: float
    retrieval_precision_at_k: float
    avg_response_time_ms: float
    evaluated_at: datetime
    details: List[Dict[str, Any]]
