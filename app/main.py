import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import Base, engine
from app.api import auth, documents, chat, search, evaluation, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("documind.main")

# Auto-create database tables on startup
Base.metadata.create_all(bind=engine)

def seed_demo_user():
    from app.core.database import SessionLocal
    from app.core.security import get_password_hash
    from app.models.domain import User
    
    db = SessionLocal()
    try:
        demo_user = db.query(User).filter(User.email == "demo@documind.ai").first()
        if not demo_user:
            hashed_pwd = get_password_hash("password123")
            demo_user = User(
                email="demo@documind.ai",
                hashed_password=hashed_pwd,
                full_name="Demo User"
            )
            db.add(demo_user)
            db.commit()
            logger.info("Created default demo user: demo@documind.ai / password123")
    except Exception as e:
        logger.error(f"Error seeding demo user: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()

seed_demo_user()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Document Intelligence & RAG Platform API",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production should restrict to allowed frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handler for clean user errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception caught on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Technical log recorded."}
    )

# Include API Routers under /api/v1
api_v1_router = FastAPI()
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(documents.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)
app.include_router(search.router, prefix=settings.API_V1_STR)
app.include_router(evaluation.router, prefix=settings.API_V1_STR)
app.include_router(health.router)

@app.get("/")
def root_redirect():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API",
        "documentation": f"{settings.API_V1_STR}/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
