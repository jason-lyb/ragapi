# app.py - config.py를 사용하여 환경변수를 로드하도록 변경
from fastapi import FastAPI, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import logging
import warnings
import asyncio
import base64
import importlib.util
from typing import Optional, List
import datetime
from fastapi import Request

# Langchain 관련 임포트
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# LLM 관련 임포트
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from models.lambda_llm import LambdaAILLM



# OpenSearch 연결을 위한 라이브러리
from opensearchpy import OpenSearch, RequestsHttpConnection

# config 파일
import config

# 로그 설정
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 오늘 날짜를 기반으로 로그 파일명 생성
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_DIR, f"rag_api_{current_date}.log")

# 로거 설정
logger = logging.getLogger("rag_api")
logger.setLevel(logging.DEBUG)

# 파일 핸들러 설정
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# 콘솔 핸들러 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 로그 포맷 설정
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# 핸들러 추가
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# OpenSearch 연결 정보
opensearch_url = config.OPENSEARCH_CONFIG["url"]
index_name = config.OPENSEARCH_CONFIG["index_name"]
username = config.OPENSEARCH_CONFIG["username"]
password = config.OPENSEARCH_CONFIG["password"]
verify_ssl = config.OPENSEARCH_CONFIG["verify_ssl"]

# ai 모델 및 api key 
gemini_model = config.GEMINI_MODEL
gemini_key = config.GEMINI_API_KEY
openai_model = config.OPENAI_MODEL
openai_key = config.OPENAI_API_KEY
lambda_model =  config.LAMBDA_MODEL_ID
lambda_key = config.LAMBDA_API_KEY

WORKERS = int(os.getenv("WORKERS", 16))
KEEPALIVE = int(os.getenv("KEEPALIVE", 65))
MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", 1000))
MAX_REQUESTS_JITTER = int(os.getenv("MAX_REQUESTS_JITTER", 50))

logger.info("API 서버 초기화 중...")

# 태그 메타데이터 정의
tags_metadata = [
    {
        "name": "검색 API",
        "description": "문서 검색 및 RAG 관련 엔드포인트",
    },
    {
        "name": "시스템",
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


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenSearch 싱글톤 클라이언트
OPENSEARCH_POOL = None
# OpenSearchVectorStore 캐싱
VECTOR_STORE = None

# Sentence Transformer 임베딩 모델 초기화
embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
logger.info(f"임베딩 모델 초기화 완료: {config.EMBEDDING_MODEL}")

# Sentence Transformer 임베딩 모델 초기화

def get_opensearch_client():
    """OpenSearch 클라이언트 싱글톤 인스턴스 반환"""
    global OPENSEARCH_POOL
    if OPENSEARCH_POOL is None:
        logger.debug(f"OpenSearch 클라이언트 생성: {opensearch_url}")
        OPENSEARCH_POOL = OpenSearch(
            hosts=[opensearch_url],
            http_auth=(username, password),
            use_ssl=verify_ssl,
            verify_certs=False,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection,
            max_retries=3,
            retry_on_timeout=True,
            timeout=30
        )
        logger.info("OpenSearch 클라이언트 생성 완료")
    return OPENSEARCH_POOL

def initialize_vectorstore():
    """OpenSearch 벡터 스토어 초기화"""
    try:
        logger.debug("OpenSearch 벡터 스토어 초기화 중...")
        # SSL 경고 억제
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vector_store = OpenSearchVectorSearch(
                index_name=index_name,
                embedding_function=embedding_model,
                opensearch_url=opensearch_url,
                http_auth=(username, password),
                verify_certs=verify_ssl,
                vector_field="vector_field",  # 벡터 필드 이름 지정
                text_field="text",            # 텍스트 필드 이름 지정
                metadata_field="metadata",    # 메타데이터 필드 이름 지정
                ssl_assert_hostname=False,    # 호스트 이름 검증 비활성화 (개발 환경용)
                ssl_show_warn=False          # SSL 경고 표시 안 함
            )
        logger.info("벡터 스토어 초기화 완료")
        return vector_store
    except Exception as e:
        logger.error(f"벡터 스토어 초기화 중 오류 발생: {e}")
        return None
    
# BaseRetriever를 상속하고 Pydantic 필드 사용
class CustomRetriever(BaseRetriever):
    documents: List[Document] = Field(default_factory=list)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 간단히 모든 문서 반환
        return self.documents

def custom_hybrid_search(client, query_text, index_name, embedding_model, top_k=5):
    """OpenSearch API를 직접 사용하여 하이브리드 검색을 수행"""
    try:
        # 시간 측정 시작
        start_time = time.time()
        logger.debug(f"하이브리드 검색 시작: 쿼리='{query_text}'")

        # 쿼리 텍스트 임베딩
        query_embedding = embedding_model.embed_query(query_text)
        
        # 하이브리드 검색 쿼리 구성 (키워드 + 벡터)
        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "text": {
                                    "query": query_text,
                                    "boost": 1.0
                                }
                            }
                        },
                        {
                            "knn": {
                                "vector_field": {
                                    "vector": query_embedding,
                                    "k": top_k,
                                    "boost": 2.0
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        # OpenSearch에 검색 요청
        logger.debug(f"OpenSearch 하이브리드 검색 실행: {index_name}, 쿼리: {query_text}")
        response = client.search(
            body=search_query,
            index=index_name
        )
        
        # 검색 결과 처리
        hits = response["hits"]["hits"]
        
        # 시간 측정 종료
        end_time = time.time()
        search_time = round((end_time - start_time) * 1000, 2)
        logger.info(f"검색시간: {search_time}ms, 검색결과 수: {len(hits)}")

        # Document 객체로 변환
        docs = []
        for i, hit in enumerate(hits):
            source = hit["_source"]
            text = source.get("text", "")
            metadata = {
                "id": hit["_id"],
                "score": hit["_score"],
                "metadata": source.get("metadata", {})
            }

            # 각 문서의 내용과 메타데이터 로깅
            logger.debug(f"문서 #{i+1} (ID: {hit['_id']}, 점수: {hit['_score']:.4f}), 내용: {text}")
            
            # 메타데이터가 있으면 로그에 기록
            if source.get("metadata"):
                logger.debug(f"메타데이터: {source.get('metadata')}")

            docs.append(Document(page_content=text, metadata=metadata))
        
        return docs
    
    except Exception as e:
        logger.error(f"하이브리드 검색 중 오류 발생: {e}")
        return []

def search_with_opensearch(request_id, client, query, search_type, top_k, index_name, embedding_model):
    """OpenSearch를 사용하여 검색을 수행하는 함수
    
    Args:
        request_id: 요청 식별자
        client: OpenSearch 클라이언트
        query: 검색 쿼리
        search_type: 검색 유형 ("hybrid", "vector", "keyword")
        top_k: 반환할 검색 결과 수
        index_name: 검색할 인덱스 이름
        embedding_model: 임베딩 모델 (벡터 검색에 사용)
        
    Returns:
        검색 결과 Document 객체 리스트
    """
    logger.debug(f"[{request_id}] {search_type} 검색 시작")
    
    if search_type == "hybrid":
        # 하이브리드 검색 (키워드 + 벡터)
        docs = custom_hybrid_search(client, query, index_name, embedding_model, top_k)
    elif search_type == "vector":
        # 순수 벡터 검색
        vector_store = initialize_vectorstore()
        docs = vector_store.similarity_search(query, k=top_k)
    elif search_type == "keyword":
        # 순수 키워드 검색
        search_query = {
            "size": top_k,
            "query": {
                "match": {
                    "text": query
                }
            }
        }
        response = client.search(
            body=search_query,
            index=index_name
        )
        
        # 검색 결과 처리
        hits = response["hits"]["hits"]
        docs = []
        for hit in hits:
            source = hit["_source"]
            text = source.get("text", "")
            metadata = {
                "id": hit["_id"],
                "score": hit["_score"],
                "metadata": source.get("metadata", {})
            }
            docs.append(Document(page_content=text, metadata=metadata))
    else:
        logger.error(f"[{request_id}] 유효하지 않은 검색 유형: {search_type}")
        raise HTTPException(status_code=400, detail=f"유효하지 않은 검색 유형: {search_type}. 'hybrid', 'vector', 또는 'keyword' 중 하나를 선택하세요.")
    
    logger.debug(f"[{request_id}] {search_type} 검색 완료: 결과 수={len(docs)}")
    return docs

# 검색된 문서를 컨텍스트로 결합
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def generate_rag_response(request_id, query, docs, llm_name):
    """
    검색된 문서를 기반으로 생성형 AI 응답을 생성합니다.
    
    Args:
        request_id (str): 요청 식별자
        query (str): 사용자 질문
        docs (List[Document]): 검색된 문서 리스트
        api_key (str): OpenAI API 키
        
    Returns:
        dict: 생성된 응답과 소스 문서를 포함하는 딕셔너리
    """
    try:
        logger.debug(f"[{request_id}] OpenAI LLM으로 응답 생성 시작")

        # 문서들을 텍스트로 변환
        context = format_docs(docs)
        logger.debug(f"[{request_id}] context : {context}")

        gen_start_time = time.time()

        # LLM 초기화
        if llm_name.lower() == "gemini":
            llm = ChatGoogleGenerativeAI(
                model=gemini_model,
                google_api_key= gemini_key,
                temperature=0.7
            )
        elif llm_name.lower() == "chatgpt":
            llm = ChatOpenAI(
                model=openai_model,
                api_key=openai_key,
                temperature=0.7
            )
        elif llm_name.lower() == "lambda":
            llm = LambdaLLM(
                model=lambda_model,
                api_key=lambda_key,
                temperature=0.7,
                max_tokens=1024
            )            
        else:
            raise ValueError(f"지원하지 않는 LLM: {llm_name}")        

        # 커스텀 프롬프트 템플릿 설정
        prompt = ChatPromptTemplate.from_template("""
        당신은 콜마너 대리운전 상담원입니다.
        제공된 정보를 바탕으로 질문에 정확하고 친절하게 답변해 주세요.
        사용자의 질문과 컨텍스트 내용이 관련이 없으면 "관련 내용 없음" 이라고 안내해줘                                                                                     
        다음은 사용자의 질문과 관련된 정보입니다:
        
        컨텍스트: {context}
        
        질문: {question}
        
        답변:
        """)            
        
        # 검색기 생성
        retriever = CustomRetriever(documents=docs)
        
        # 현대적인 LangChain 체인 구성 방식 (LCEL - LangChain Expression Language)
        rag_chain = (
            {
                "context": RunnableLambda(lambda x: format_docs(docs)),
                "question": RunnableLambda(lambda x: x["question"])
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        result = rag_chain.invoke({"context": context, "question": query})
        gen_end_time = time.time()
        gen_duration = round((gen_end_time - gen_start_time) * 1000, 2)
        logger.info(f"[{request_id}] 응답 생성 완료: 소요시간={gen_duration}ms")
        
        # 결과 형식에 따라 적절히 처리
        if isinstance(result, dict) and "answer" in result:
            answer_text = result["answer"]
            logger.debug(f"[{request_id}] 생성된 응답: {answer_text}")
            return {
                "answer": answer_text,
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in docs
                ]
            }
        elif isinstance(result, str):
            # 결과가 문자열인 경우
            logger.debug(f"[{request_id}] 생성된 응답: {result}")
            return {
                "answer": result,
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in docs
                ]
            }
        else:
            # 다른 형식의 결과인 경우
            logger.debug(f"[{request_id}] 생성된 응답 (형식: {type(result)}): {str(result)}")
            # 적절한 형태로 반환하기 위해 결과 구조 파악
            if hasattr(result, "result"):
                answer = result.result
            elif hasattr(result, "output_text"):
                answer = result.output_text
            else:
                answer = str(result)
                
            return {
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in docs
                ]
            }
    except Exception as e:
        logger.error(f"[{request_id}] 생성형 응답 중 오류 발생: {e}")
        # OpenAI 오류가 발생한 경우 검색 결과만 반환
        return {
            "answer": f"생성형 응답 중 오류가 발생했습니다: {str(e)}",
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs
            ]
        }

async def analyze_image_url_with_model(request_id, image_url, img_base64, description, model_type):
    """URL을 통해 특정 모델을 사용하여 이미지를 분석합니다.
    
    Args:
        request_id: 요청 ID
        image_url: 이미지 URL
        img_base64: 이미지의 base64 인코딩 (Gemini용)
        description: 이미지에 대한 추가 설명
        model_type: 사용할 모델 타입 ('chatgpt' 또는 'gemini')
        
    Returns:
        분석 결과를 포함한 딕셔너리
    """
    
    logger.info(f"[{request_id}] {model_type} 모델로 URL 이미지 분석 시작: {image_url}")
    
    start_time = time.time()
    
    # 이미지 분석 프롬프트 구성
    prompt = "이 이미지를 분석하고 상세히 설명해주세요."
    if description:
        prompt += f" 특히 다음 내용에 주목해주세요: {description}"
    
    try:
        # 모델 유형에 따른 이미지 분석
        if model_type.lower() == "chatgpt":
            # OpenAI의 GPT-4 Vision 사용 (URL 직접 전달)
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            
            response = client.chat.completions.create(
                model=openai_model,  # 또는 최신 비전 모델
                messages=[
                    {"role": "system", "content": "당신은 이미지를 분석하는 전문가입니다. 이미지의 내용을 정확하고 상세하게 설명해주세요."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]}
                ],
                max_tokens=1000
            )
            
            analysis = response.choices[0].message.content
            model_info = openai_model
            
        elif model_type.lower() == "gemini":
            # Google의 Gemini Pro Vision 사용 (URL 직접 전달)
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            
            genai.configure(api_key=gemini_key)
            
            # 안전 설정
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
            
            # Gemini Pro Vision 모델
            model = genai.GenerativeModel(
                model_name=gemini_model,
                safety_settings=safety_settings
            )
            
            # base64 인코딩된 이미지 사용
            response = model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": img_base64}
            ])
            
            analysis = response.text
            model_info = gemini_model
            
        else:
            raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        logger.info(f"[{request_id}] {model_type} URL 이미지 분석 완료: 소요시간={processing_time}ms")
        
        return {
            "analysis": analysis,
            "model": model_info,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.exception(f"[{request_id}] {model_type} URL 이미지 분석 중 오류: {e}")
        raise Exception(f"{model_type} URL 이미지 분석 중 오류: {str(e)}")


# JSON 요청을 위한 Pydantic 모델
class SearchRequest(BaseModel):
    query: str
    search_type: str = "hybrid"
    llm_type: str = "chatgpt"
    top_k: int = 3

@app.post("/search", tags=["검색 API"])
async def search(
    query: str = Form(...),
    search_type: str = Form("hybrid"),
    llm_type: str = Form("chatgpt"),
    top_k: int = Form(3)
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
        result = generate_rag_response(request_id, query, docs, llm_type)
        
        total_time = round((time.time() - search_start_time) * 1000, 2)
        logger.info(f"[{request_id}] 전체 처리 완료: 총 소요시간={total_time}ms")
        
        return result
    
    except Exception as e:
        logger.exception(f"[{request_id}] 검색 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")
    
@app.post("/search-json", tags=["검색 API"])
async def search_json(request: SearchRequest):

    """문서에서 쿼리를 검색하고 결과를 반환합니다.
    
    - search_type: 검색 유형 ("hybrid", "vector", "keyword")
    - llm_type: GTP 유형 ("chatgpt", "gemini", "all(비교)")
    - top_k: 반환할 검색 결과 수
    """

    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] JSON 검색 요청 수신: 쿼리='{request.query}', 검색유형={request.search_type}, top_k={request.top_k}")

    query = request.query
    search_type = request.search_type
    llm_type = request.llm_type
    top_k = request.top_k
    
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
            logger.info(f"[{request_id}] 'all' 모드: OpenAI와 Gemini 모두 호출")
            # OpenAI 결과 생성
            openai_result = generate_rag_response(request_id, query, docs, "chatgpt")
            # Gemini 결과 생성
            gemini_result = generate_rag_response(request_id, query, docs, "gemini")
            
            # 두 결과 합치기
            combined_result = {
                "chatgpt": {
                    "answer": openai_result["answer"],
                    "model": "chatgpt"
                },
                "gemini": {
                    "answer": gemini_result["answer"],
                    "model": "gemini"
                },
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in docs
                ]
            }
            
            total_time = round((time.time() - search_start_time) * 1000, 2)
            logger.info(f"[{request_id}] 'all' 모드 처리 완료: 총 소요시간={total_time}ms")
            
            return combined_result
        else:        
            result = generate_rag_response(request_id, query, docs, llm_type)
        
        total_time = round((time.time() - search_start_time) * 1000, 2)
        answer_preview = result["answer"]
        logger.info(f"[{request_id}] 전체 처리 완료: 총 소요시간={total_time}ms, 응답={answer_preview}")
                
        return result
    
    except Exception as e:
        logger.exception(f"[{request_id}] 검색 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")

@app.post("/analyze-image-url", tags=["이미지 분석 API"])
async def analyze_image_url(
    request: Request,
    url: str = Form(...),
    description: str = Form(None),
    llm_type: str = Form("chatgpt")
):
    """URL에서 이미지를 분석하고 결과를 반환합니다.
    
    - url: 분석할 이미지의 URL
    - description: 이미지에 대한 추가 설명 (선택사항)
    - llm_type: 사용할 LLM 유형 ("chatgpt", "gemini", "all(비교)")
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
            logger.info(f"[{request_id}] 'all' 모드: OpenAI와 Gemini 모두 호출")
            
            # OpenAI 이미지 분석 (URL 직접 전달)
            openai_result = await analyze_image_url_with_model(request_id, url, img_base64, description, "chatgpt")
            
            # Gemini 이미지 분석 (base64 인코딩 사용)
            gemini_result = await analyze_image_url_with_model(request_id, url, img_base64, description, "gemini")
            
            # 두 결과 합치기
            combined_result = {
                "chatgpt": {
                    "analysis": openai_result["analysis"],
                    "model": "chatgpt"
                },
                "gemini": {
                    "analysis": gemini_result["analysis"],
                    "model": "gemini"
                },
                "image_info": {
                    "filename": file_name,
                    "url": url,
                    "size": file_size,
                    "format": img_format
                }
            }
            
            total_time = round((time.time() - analysis_start_time) * 1000, 2)
            logger.info(f"[{request_id}] 'all' 모드 URL 이미지 분석 완료: 총 소요시간={total_time}ms")
            
            return combined_result
        else:
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

      
@app.get("/health", tags=["시스템"])
def health():
    logger.debug("헬스 체크 요청")
    return {"status": "normal"}

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