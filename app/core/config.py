import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "DocuMind AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "documind-super-secret-key-change-in-production-2025")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Database (Defaults to SQLite for instant local dev, easily overridden by postgresql://... in docker)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./documind.db")
    
    # Storage Paths
    BASE_DATA_DIR: str = os.getenv("BASE_DATA_DIR", "./data")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "./data/vector_store")
    
    # RAG & Chunking Parameters
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "150"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    
    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")  # Options: mock, gemini, openai, ollama
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure data directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
