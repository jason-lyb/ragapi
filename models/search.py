from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SearchRequest(BaseModel):
    query: str
    search_type: str = Field(default="hybrid", description="검색 유형: 'hybrid', 'vector', 'keyword'")
    llm_type: str = Field(default="chatgpt", description="LLM 유형: 'chatgpt', 'gemini', 'all'")
    top_k: int = Field(default=5, ge=1, le=20, description="반환할 결과 수")
    user_id: str
    
class SearchSource(BaseModel):
    content: str
    metadata: Dict[str, Any]
    
class SearchResponse(BaseModel):
    answer: str
    sources: List[SearchSource] = []

class CombinedSearchResponse(BaseModel):
    chatgpt: Dict[str, Any]
    gemini: Dict[str, Any]
    sources: List[SearchSource] = []