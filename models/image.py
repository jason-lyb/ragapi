from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ImageAnalysisRequest(BaseModel):
    url: Optional[str] = None
    description: Optional[str] = None
    llm_type: str = Field(default="chatgpt", description="LLM 유형: 'chatgpt', 'gemini', 'all'")

class ImageInfo(BaseModel):
    filename: str
    url: Optional[str] = None
    size: Optional[int] = None
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    exif: Optional[Dict[str, Any]] = None
    
class ImageAnalysisResponse(BaseModel):
    analysis: str
    model: str
    processing_time_ms: float
    image_info: ImageInfo

class CombinedImageAnalysisResponse(BaseModel):
    chatgpt: Dict[str, Any]
    gemini: Dict[str, Any]
    image_info: ImageInfo