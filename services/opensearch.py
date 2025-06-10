# OpenSearch 연결을 위한 라이브러리
from opensearchpy import OpenSearch, RequestsHttpConnection
import time
from config import OPENSEARCH_CONFIG, EMBEDDING_MODEL
from logger import logger

from langchain.schema import Document

# OpenSearch 연결 정보
opensearch_url = OPENSEARCH_CONFIG["url"]
index_name = OPENSEARCH_CONFIG["index_name"]
username = OPENSEARCH_CONFIG["username"]
password = OPENSEARCH_CONFIG["password"]
verify_ssl = OPENSEARCH_CONFIG["verify_ssl"]

# OpenSearch 싱글톤 클라이언트
OPENSEARCH_POOL = None
# OpenSearchVectorStore 캐싱
VECTOR_STORE = None

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
                embedding_function=EMBEDDING_MODEL,
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