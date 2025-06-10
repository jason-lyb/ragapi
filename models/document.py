from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DocumentBase(BaseModel):
    document_id: str
    title: str
    filename: str
    description: Optional[str] = ""
    tags: List[str] = []
    upload_time: float
    file_type: str
    chunks: int

class DocumentResponse(DocumentBase):
    """개별 문서 응답 모델"""
    pass

class DocumentListResponse(BaseModel):
    """문서 목록 응답 모델"""
    documents: List[DocumentBase]
    total: int
    page: int
    limit: int

class DocumentIndexResponse(BaseModel):
    """문서 인덱싱 응답 모델"""
    document_id: str
    title: str
    filename: str
    chunks: int
    processing_time_ms: float

# 삭제 관련 모델
class DeleteResponse(BaseModel):
    """삭제 응답 모델"""
    success: bool
    message: str
    deleted_count: int
    deleted_ids: List[str] = []

class BatchDeleteRequest(BaseModel):
    """일괄 삭제 요청 모델"""
    document_ids: List[str]