# app.py - config.py를 사용하여 환경변수를 로드하도록 변경
from fastapi import FastAPI, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import asyncio

from logger import logger
from config import EMBEDDING_MODEL, TEMP_DIR

logger.info("API 서버 초기화 중...")

# 내부 모듈 가져오기
from api.search import router as search_router
from api.image_analysis import router as image_analysis_router
from api.document import router as document_router
from api.health import router as health_router

WORKERS = int(os.getenv("WORKERS", 16))
KEEPALIVE = int(os.getenv("KEEPALIVE", 65))
MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", 1000))
MAX_REQUESTS_JITTER = int(os.getenv("MAX_REQUESTS_JITTER", 50))

# 태그 메타데이터 정의
tags_metadata = [
    {
        "name": "검색 API",
        "description": "문서 검색 및 RAG 관련 엔드포인트",
    },
    {
        "name": "이미지 분석 API",
        "description": "이미지 분석 엔드포인트",
    },
    {
        "name": "문서 API",
        "description": "문서 업로드, 조회, 삭제 엔드포인트",
    },   
    {
        "name": "시스템 상태",
        "description": "시스템 상태 및 관리 엔드포인트",
    }
]

app = FastAPI(
    title="콜마너 RAG 검색 API", 
    description="Langchain과 OpenSearch를 활용한 한국어 문서 기반 RAG 검색 API",
    openapi_tags=tags_metadata,
    version="1.0.0",
    contact={
        "name": "솔루션개발팀",
        "email": "jason.lyb@cmnp.co.kr",
    },
    license_info={
        "name": "CMNP",
        "url": "https://www.callmaner.com",
    }
)

# 임시 디렉토리 생성
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(search_router, prefix="/api", tags=["검색 API"])
app.include_router(image_analysis_router, prefix="/api", tags=["이미지 분석 API"])
app.include_router(document_router, prefix="/api", tags=["문서 API"])
app.include_router(health_router, prefix="/api", tags=["시스템 상태"])  # health 라우터 추가 (접두사 없음)

if __name__ == "__main__":
    import hypercorn.asyncio
    from hypercorn.config import Config
    
    # 서버 시작 로그
    logger.info(f"서버 시작: 작업자={WORKERS}, keepalive={KEEPALIVE}, max_requests={MAX_REQUESTS}")
    
    config = Config()
    config.bind = ["0.0.0.0:8002"]
#    config.workers = WORKERS
    config.keep_alive_timeout = KEEPALIVE
    config.max_requests = MAX_REQUESTS
    config.max_requests_jitter = MAX_REQUESTS_JITTER
    config.worker_class = "uvloop"
    
    logger.info("Hypercorn 서버 시작 중...")
    asyncio.run(hypercorn.asyncio.serve(app, config))