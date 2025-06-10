from fastapi import APIRouter, HTTPException, File, Form, UploadFile, Request
import time
import os
import base64
from models.image import ImageAnalysisRequest, ImageAnalysisResponse, CombinedImageAnalysisResponse
from services.llm import analyze_image_url_with_model
from services.image import fetch_image_metadata, download_image, extract_image_metadata
from logger import logger
from config import TEMP_DIR, OPENAI_MODEL, GEMINI_MODEL, LAMBDA_MODEL, LAMBDA_IMG_MODEL
import asyncio

router = APIRouter()

async def safe_analyze_with_model(request_id: str, url: str, img_base64: str, description: str, model_type: str):
    """안전하게 모델 분석을 수행하고 오류시 오류 정보를 반환"""
    try:
        result = await analyze_image_url_with_model(request_id, url, img_base64, description, model_type)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"[{request_id}] {model_type} 모델 분석 실패: {e}")
        return {
            "success": False, 
            "error": str(e),
            "error_type": type(e).__name__
        }

@router.post("/analyze-image-url", tags=["이미지 분석 API"])
async def analyze_image_url(
    request: Request,
    url: str = Form(...),
    description: str = Form(None),
    llm_type: str = Form("chatgpt")
):
    """URL에서 이미지를 분석하고 결과를 반환합니다.
    
    - url: 분석할 이미지의 URL
    - description: 이미지에 대한 추가 설명 (텍스트를 추출하고 일자와 주소를 추출해줘)
    - llm_type: 사용할 LLM 유형 ("chatgpt", "gemini", "llama",  "all(비교)")
    """
    
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] URL 이미지 분석 요청 수신: URL='{url}', LLM 유형={llm_type}")
    
    try:
        # URL 유효성 검사
        if not url:
            raise HTTPException(status_code=400, detail="이미지 URL이 필요합니다")
            
        # URL 형식 확인
        if not url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="올바른 URL 형식이 아닙니다 (http:// 또는 https:// 포함)")
        
        # 이미지 메타데이터 추출 (URL 정보)
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path) or "image_from_url"
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # 헤더 정보 확인 (선택 사항)
        img_format = 'unknown'
        file_size = None
        
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.head(url, timeout=5) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        if content_type.startswith('image/'):
                            img_format = content_type.split('/')[1].lower()
                            if img_format == 'jpeg':
                                img_format = 'jpg'
                        
                        content_length = response.headers.get('content-length')
                        if content_length and content_length.isdigit():
                            file_size = int(content_length)
        except Exception as e:
            logger.warning(f"[{request_id}] 이미지 헤더 확인 중 오류: {e}")
            if file_ext:
                img_format = file_ext.replace('.', '')
        
        # Gemini API를 위해 이미지 데이터 획득 (base64)
        img_base64 = None
        if llm_type.lower() in ["gemini", "all"]:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            image_data = await response.read()
                            img_base64 = base64.b64encode(image_data).decode('utf-8')
                        else:
                            logger.error(f"[{request_id}] 이미지 다운로드 실패: 상태 코드 {response.status}")
                            raise HTTPException(status_code=400, detail=f"이미지를 다운로드할 수 없습니다. 상태 코드: {response.status}")
            except Exception as e:
                logger.error(f"[{request_id}] 이미지 다운로드 중 오류: {e}")
                raise HTTPException(status_code=500, detail=f"이미지 다운로드 중 오류: {str(e)}")
        
        # 이미지 분석 시작
        analysis_start_time = time.time()
        
        # llm_type에 따른 이미지 분석 실행
        if llm_type.lower() == "all":
            logger.info(f"[{request_id}] 'all' 모드: OpenAI, Gemini, Llama 모두 호출")
            
            # 세 API 호출을 병렬로 처리 (안전한 방식으로)
            openai_task = safe_analyze_with_model(request_id, url, img_base64, description, "chatgpt")
            gemini_task = safe_analyze_with_model(request_id, url, img_base64, description, "gemini")
            llama_task = safe_analyze_with_model(request_id, url, img_base64, description, "llama")
            
            # 모든 작업을 동시에 실행하고 결과를 기다림
            openai_result, gemini_result, llama_result = await asyncio.gather(
                openai_task, gemini_task, llama_task, return_exceptions=True
            )
            
            # 결과 처리 및 조합 - 동일한 JSON 구조 유지
            combined_result = {
                "chatgpt": {
                    "status": "success",
                    "analysis": "",
                    "model": OPENAI_MODEL
                },
                "gemini": {
                    "status": "success", 
                    "analysis": "",
                    "model": GEMINI_MODEL
                },
                "llama": {
                    "status": "success",
                    "analysis": "",
                    "model": LAMBDA_IMG_MODEL
                },
                "summary": {
                    "total_models": 3,
                    "successful_models": 0,
                    "failed_models": 0,
                    "success_rate": 0.0
                },
                "image_info": {
                    "filename": file_name,
                    "url": url,
                    "size": file_size,
                    "format": img_format
                }
            }
            
            # OpenAI 결과 처리
            if isinstance(openai_result, dict):
                if openai_result.get("success"):
                    combined_result["chatgpt"]["analysis"] = openai_result["data"]["analysis"]
                    combined_result["summary"]["successful_models"] += 1
                else:
                    combined_result["chatgpt"]["status"] = "failed"
                    combined_result["chatgpt"]["analysis"] = f"오류 발생: {openai_result.get('error', 'Unknown error')}"
                    combined_result["summary"]["failed_models"] += 1
            else:
                # Exception이 직접 반환된 경우
                combined_result["chatgpt"]["status"] = "failed"
                combined_result["chatgpt"]["analysis"] = f"오류 발생: {str(openai_result)}"
                combined_result["summary"]["failed_models"] += 1
            
            # Gemini 결과 처리
            if isinstance(gemini_result, dict):
                if gemini_result.get("success"):
                    combined_result["gemini"]["analysis"] = gemini_result["data"]["analysis"]
                    combined_result["summary"]["successful_models"] += 1
                else:
                    combined_result["gemini"]["status"] = "failed"
                    combined_result["gemini"]["analysis"] = f"오류 발생: {gemini_result.get('error', 'Unknown error')}"
                    combined_result["summary"]["failed_models"] += 1
            else:
                combined_result["gemini"]["status"] = "failed"
                combined_result["gemini"]["analysis"] = f"오류 발생: {str(gemini_result)}"
                combined_result["summary"]["failed_models"] += 1
            
            # Llama 결과 처리
            if isinstance(llama_result, dict):
                if llama_result.get("success"):
                    combined_result["llama"]["analysis"] = llama_result["data"]["analysis"]
                    combined_result["summary"]["successful_models"] += 1
                else:
                    combined_result["llama"]["status"] = "failed"
                    combined_result["llama"]["analysis"] = f"오류 발생: {llama_result.get('error', 'Unknown error')}"
                    combined_result["summary"]["failed_models"] += 1
            else:
                combined_result["llama"]["status"] = "failed"
                combined_result["llama"]["analysis"] = f"오류 발생: {str(llama_result)}"
                combined_result["summary"]["failed_models"] += 1
            
            # 성공률 계산
            combined_result["summary"]["success_rate"] = round(
                (combined_result["summary"]["successful_models"] / combined_result["summary"]["total_models"]) * 100, 2
            )
            
            total_time = round((time.time() - analysis_start_time) * 1000, 2)
            
            # 모든 모델이 실패한 경우에만 HTTP 오류 반환
            if combined_result["summary"]["successful_models"] == 0:
                logger.error(f"[{request_id}] 모든 모델 분석 실패")
                raise HTTPException(
                    status_code=500, 
                    detail="모든 모델에서 이미지 분석에 실패했습니다. 결과를 확인해주세요."
                )
            
            logger.info(f"[{request_id}] 'all' 모드 URL 이미지 분석 완료: "
                       f"성공={combined_result['summary']['successful_models']}, "
                       f"실패={combined_result['summary']['failed_models']}, "
                       f"총 소요시간={total_time}ms")
            
            return combined_result
        else:
            # 단일 모델 분석
            result = await analyze_image_url_with_model(request_id, url, img_base64, description, llm_type)
            
            # 이미지 URL 정보 추가
            result["image_info"] = {
                "url": url,
                "filename": file_name,
                "size": file_size,
                "format": img_format
            }
            
            total_time = round((time.time() - analysis_start_time) * 1000, 2)
            logger.info(f"[{request_id}] URL 이미지 분석 완료: 총 소요시간={total_time}ms")
            
            return result
            
    except Exception as e:
        logger.exception(f"[{request_id}] URL 이미지 분석 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"URL 이미지 분석 중 오류 발생: {str(e)}")