from fastapi import APIRouter, HTTPException, Depends, Form
import time
from opensearchpy import OpenSearch
from models.search import SearchRequest, SearchResponse, CombinedSearchResponse
from services.opensearch import get_opensearch_client, search_with_opensearch
from services.llm import generate_rag_response
from logger import logger
from config import OPENSEARCH_CONFIG, EMBEDDING_MODEL, GEMINI_MODEL, OPENAI_MODEL, LAMBDA_MODEL
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio

router = APIRouter()

index_name = OPENSEARCH_CONFIG["index_name"]

# Sentence Transformer 임베딩 모델 초기화
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
logger.info(f"임베딩 모델 초기화 완료: {EMBEDDING_MODEL}")

@router.post("/search", tags=["검색 API"])
async def search(
    query: str = Form(...),
    search_type: str = Form("hybrid"),
    llm_type: str = Form("chatgpt"),
    top_k: int = Form(3),
    user_id: str = Form(...)
):

    """문서에서 쿼리를 검색하고 결과를 반환합니다.
    
    - search_type: 검색 유형 ("hybrid", "vector", "keyword")
    - llm_type: GTP 유형 ("chatgpt", "gemini")
    - top_k: 반환할 검색 결과 수
    """

    # 필수 파라미터 확인
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] 검색 요청 수신: 쿼리='{query}', 검색유형={search_type}, top_k={top_k}")
    
    try:
        # OpenSearch 클라이언트 가져오기
        client = get_opensearch_client()
        logger.debug(f"[{request_id}] OpenSearch 클라이언트 획득")
        
        # 인덱스가 있는지 확인
        if not client.indices.exists(index=index_name):
            logger.error(f"[{request_id}] 인덱스 '{index_name}' 없음")
            raise HTTPException(status_code=400, detail=f"인덱스 '{index_name}'가 없습니다. 먼저 문서를 업로드하세요.")
        
        # 검색 유형에 따라 다른 검색 방법 사용
        logger.info(f"[{request_id}] 검색 시작: 유형={search_type}, LLM={llm_type}, 쿼리={query}")
        search_start_time = time.time()

        # 검색 함수 호출
        docs = search_with_opensearch(request_id, client, query, search_type, top_k, index_name, embedding_model)
        
        search_duration = round((time.time() - search_start_time) * 1000, 2)
        logger.info(f"[{request_id}] 검색 완료: 소요시간={search_duration}ms, 결과 수={len(docs)}")
        
        # 검색 결과가 없으면 빈 응답
        if not docs:
            logger.warning(f"[{request_id}] 검색 결과 없음: 쿼리={query}")
            return {
                "answer": "검색 결과가 없습니다. 다른 검색어로 시도해보세요.",
                "sources": []
            }
        
        # RAG를 통한 생성형 응답 함수 호출
        result = generate_rag_response(request_id, query, docs, llm_type, user_id)
        
        total_time = round((time.time() - search_start_time) * 1000, 2)
        logger.info(f"[{request_id}] 전체 처리 완료: 총 소요시간={total_time}ms")
        
        return result
    
    except Exception as e:
        logger.exception(f"[{request_id}] 검색 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")

@router.post("/search-json", response_model=None, tags=["검색 API"])
async def search_json(request: SearchRequest):

    """문서에서 쿼리를 검색하고 결과를 반환합니다.
    
    - search_type: 검색 유형 ("hybrid", "vector", "keyword")
    - llm_type: GTP 유형 ("chatgpt", "gemini", "llama", "all(비교)")
    - top_k: 반환할 검색 결과 수
    - user_id: 사용자 id (대화히스토리 유지위해)
    """

    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] JSON 검색 요청 수신: 쿼리='{request.query}', 검색유형={request.search_type}, top_k={request.top_k}")

    query = request.query
    search_type = request.search_type
    llm_type = request.llm_type
    top_k = request.top_k
    user_id = request.user_id
    
    # 필수 파라미터 확인
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] 검색 요청 수신: 쿼리='{query}', 검색유형={search_type}, top_k={top_k}")
    
    try:
        # OpenSearch 클라이언트 가져오기
        client = get_opensearch_client()
        logger.debug(f"[{request_id}] OpenSearch 클라이언트 획득")
        
        # 인덱스가 있는지 확인
        if not client.indices.exists(index=index_name):
            logger.error(f"[{request_id}] 인덱스 '{index_name}' 없음")
            raise HTTPException(status_code=400, detail=f"인덱스 '{index_name}'가 없습니다. 먼저 문서를 업로드하세요.")
        
        # 검색 유형에 따라 다른 검색 방법 사용
        logger.info(f"[{request_id}] 검색 시작: 유형={search_type}, LLM={llm_type}, 쿼리={query}")
        search_start_time = time.time()

        # 검색 함수 호출
        docs = search_with_opensearch(request_id, client, query, search_type, top_k, index_name, embedding_model)
        
        search_duration = round((time.time() - search_start_time) * 1000, 2)
        logger.info(f"[{request_id}] 검색 완료: 소요시간={search_duration}ms, 결과 수={len(docs)}")
        
        # 검색 결과가 없으면 빈 응답
        if not docs:
            logger.warning(f"[{request_id}] 검색 결과 없음: 쿼리={query}")
            return {
                "answer": "검색 결과가 없습니다. 다른 검색어로 시도해보세요.",
                "sources": []
            }
        
        # RAG를 통한 생성형 응답 함수 호출
        # llm_type이 'all'인 경우 두 모델 모두 호출
        if llm_type.lower() == "all":
            logger.info(f"[{request_id}] 'all' 모드: OpenAI, Gemini, llama 모두 호출")

            # OpenAI 결과 생성
#            openai_result = await generate_rag_response(request_id, query, docs, "chatgpt")
            # Gemini 결과 생성
#            gemini_result = await generate_rag_response(request_id, query, docs, "gemini")

            # 두 API 호출을 병렬로 처리
            openai_task = generate_rag_response(request_id, request.query, docs, "chatgpt", user_id)
            gemini_task = generate_rag_response(request_id, request.query, docs, "gemini", user_id)
            llama_task = generate_rag_response(request_id, request.query, docs, "llama", user_id)
            
            # 두 작업을 동시에 실행하고 결과를 기다림
            openai_result, gemini_result, llama_result = await asyncio.gather(openai_task, gemini_task, llama_task)
 
            logger.info(f"[{request_id}] 'chatgpt Result = {openai_result["answer"]}")
            logger.info(f"[{request_id}] 'gemini Result = {gemini_result["answer"]}")
            logger.info(f"[{request_id}] 'llama Result = {llama_result["answer"]}")

            # 소스 변환
            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs
            ]

            # 두 결과 합치기
            combined_result = {
                "chatgpt": {
                    "answer": openai_result["answer"],
                    "model": OPENAI_MODEL
                },
                "gemini": {
                    "answer": gemini_result["answer"],
                    "model": GEMINI_MODEL
                },
                "llama": {
                    "answer": llama_result["answer"],
                    "model": LAMBDA_MODEL
                },                
                "sources": sources
            }
            
            total_time = round((time.time() - search_start_time) * 1000, 2)
            logger.info(f"[{request_id}] 'all' 모드 처리 완료: 총 소요시간={total_time}ms")
            
            return combined_result
        else:        
            result = await generate_rag_response(request_id, query, docs, llm_type, user_id)
        
        total_time = round((time.time() - search_start_time) * 1000, 2)
        answer_preview = result["answer"]
        logger.info(f"[{request_id}] 전체 처리 완료: 총 소요시간={total_time}ms, 응답={answer_preview}")
                
        return result
    
    except Exception as e:
        logger.exception(f"[{request_id}] 검색 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")