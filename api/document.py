from fastapi import APIRouter, HTTPException, Query, Form, Path
from typing import List, Optional, Dict, Any
import time
import asyncio
from opensearchpy import OpenSearch
from models.document import DeleteResponse, BatchDeleteRequest

from logger import logger
from config import OPENSEARCH_CONFIG

INDEX_NAME = OPENSEARCH_CONFIG["index_name"]

router = APIRouter()

@router.get("/search", tags=["문서 API"])
async def search_documents(
    query: str,
    search_type: str = Query("hybrid", description="검색 유형: hybrid, vector, keyword"),
    category: Optional[str] = Query(None, description="카테고리 필터"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    size: int = Query(10, ge=1, le=100, description="페이지당 결과 수")
):
    """문서를 검색합니다.
    
    Args:
        query: 검색어
        search_type: 검색 유형 (hybrid, vector, keyword)
        category: 카테고리 필터
        page: 페이지 번호
        size: 페이지당 결과 수
    
    Returns:
        검색 결과 목록
    """
    
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] 검색 요청: 쿼리='{query}', 유형={search_type}, 카테고리={category}")
    
    try:
        # OpenSearch 클라이언트 가져오기
        from services.opensearch import get_opensearch_client
        client = get_opensearch_client()
        
        # 인덱스가 없으면 오류
        if not client.indices.exists(index=INDEX_NAME):
            logger.error(f"[{request_id}] 인덱스 '{INDEX_NAME}'가 없습니다")
            return {"results": [], "total": 0, "page": page, "size": size}
        
        # 페이징 처리
        from_idx = (page - 1) * size
        
        # 벡터 임베딩 생성 (벡터 검색 또는 하이브리드 검색인 경우)
        if search_type in ["vector", "hybrid"]:
            from services.document_processor import embeddings
            import asyncio
            vector_embedding = await asyncio.to_thread(embeddings.embed_query, query)
        
        # 검색 쿼리 구성
        search_query = {
            "from": from_idx,
            "size": size,
            "track_total_hits": True
        }
        
        # 검색 유형에 따른 쿼리 구성
        if search_type == "keyword":
            # 키워드 검색
            search_query["query"] = {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "text": query
                            }
                        }
                    ]
                }
            }
        elif search_type == "vector":
            # 벡터 검색
            search_query["query"] = {
                "knn": {
                    "vector_field": {
                        "vector": vector_embedding,
                        "k": size
                    }
                }
            }
        else:  # hybrid
            from services.document_processor import embeddings
            # 하이브리드 검색 (키워드 + 벡터)
            query_embedding = embeddings.embed_query(query)
            
            # 하이브리드 검색 쿼리 구성 (키워드 + 벡터)
            search_query = {
                "size": 5,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "boost": 1.0
                                    }
                                }
                            },
                            {
                                "knn": {
                                    "vector_field": {
                                        "vector": query_embedding,
                                        "k": 5,
                                        "boost": 2.0
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        
        # 카테고리 필터 추가
        if category:
            if "query" in search_query and "bool" in search_query["query"]:
                # bool 쿼리가 이미 있는 경우 filter 추가
                search_query["query"]["bool"].setdefault("filter", []).append({
                    "term": {"metadata.category.keyword": category}
                })
            elif "query" in search_query:
                # 쿼리는 있지만 bool이 없는 경우
                original_query = search_query["query"]
                search_query["query"] = {
                    "bool": {
                        "must": [original_query],
                        "filter": [{"term": {"metadata.category.keyword": category}}]
                    }
                }
            else:
                # 쿼리가 없는 경우 (vector 검색)
                search_query["post_filter"] = {
                    "term": {"metadata.category.keyword": category}
                }
        
        # 검색 실행
        start_time = time.time()
        response = client.search(
            index=INDEX_NAME,
            body=search_query
        )
        search_time = round((time.time() - start_time) * 1000, 2)
        
        # 결과 처리
        hits = response["hits"]["hits"]
        total_hits = response["hits"]["total"]["value"]
        
        results = []
        for hit in hits:
            source = hit["_source"]
            metadata = source.get("metadata", {})
            
            result_item = {
                "id": hit["_id"],
                "score": hit.get("_score", 0),
                "text": source.get("text", ""),
                "source_url": metadata.get("source_url", ""),
                "category": metadata.get("category", ""),
                "chunk_id": metadata.get("chunk_id", 0),
                "created_at": metadata.get("created_at", "")
            }
            
            results.append(result_item)
        
        logger.info(f"[{request_id}] 검색 완료: 소요시간={search_time}ms, 총 결과={total_hits}, 반환 결과={len(results)}")
        
        return {
            "results": results,
            "total": total_hits,
            "page": page,
            "size": size,
            "search_time_ms": search_time
        }
        
    except Exception as e:
        logger.exception(f"[{request_id}] 검색 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")


@router.get("/documents/sources", tags=["문서 API"])
async def list_document_sources(
    category: Optional[str] = Query(None, description="특정 카테고리로 필터링"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    size: int = Query(10, ge=1, le=100, description="페이지당 결과 수")
):
    """문서 소스 URL 목록을 가져옵니다.
    
    Args:
        category: 카테고리 필터 (app_usage, products)
        page: 페이지 번호
        size: 페이지당 결과 수
    
    Returns:
        문서 소스 URL 목록
    """
    
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] 문서 소스 목록 요청: 카테고리={category}, 페이지={page}, 크기={size}")
    
    try:
        # OpenSearch 클라이언트 가져오기
        from services.opensearch import get_opensearch_client
        client = get_opensearch_client()
        
        # 인덱스가 없으면 오류
        if not client.indices.exists(index=INDEX_NAME):
            logger.error(f"[{request_id}] 인덱스 '{INDEX_NAME}'가 없습니다")
            return {"sources": [], "total": 0, "page": page, "size": size}
        
        # 쿼리 구성
        aggs_query = {
            "size": 0,
            "aggs": {
                "sources": {
                    "terms": {
                        "field": "metadata.source_url.keyword",
                        "size": 10000  # 최대 소스 수
                    },
                    "aggs": {
                        "top_document": {
                            "top_hits": {
                                "size": 1,
                                "sort": [{"metadata.chunk_id": {"order": "asc"}}],
                                "_source": ["metadata", "text"]
                            }
                        },
                        "chunk_count": {
                            "value_count": {
                                "field": "_id"
                            }
                        }
                    }
                }
            }
        }
        
        # 카테고리 필터 추가
        if category:
            aggs_query["query"] = {
                "term": {
                    "metadata.category.keyword": category
                }
            }
        
        # 쿼리 실행
        start_time = time.time()
        response = client.search(
            index=INDEX_NAME,
            body=aggs_query
        )
        query_time = round((time.time() - start_time) * 1000, 2)
        
        # 결과 처리
        buckets = response["aggregations"]["sources"]["buckets"]
        total_sources = len(buckets)
        
        # 페이징 처리
        start_idx = (page - 1) * size
        end_idx = min(start_idx + size, total_sources)
        page_buckets = buckets[start_idx:end_idx]
        
        sources = []
        for bucket in page_buckets:
            source_url = bucket["key"]
            doc_count = bucket["chunk_count"]["value"]
            
            # 메타데이터 가져오기
            top_hit = bucket["top_document"]["hits"]["hits"][0]["_source"]
            metadata = top_hit.get("metadata", {})
            text = top_hit.get("text", "")  # 텍스트 추출            

            source_item = {
                "source_url": source_url,
                "category": metadata.get("category", ""),
                "created_at": metadata.get("created_at", ""),
                "chunk_count": doc_count,
                "text": text  # 텍스트 미리보기 추가
            }
            
            sources.append(source_item)
        
        logger.info(f"[{request_id}] 문서 소스 목록 완료: 소요시간={query_time}ms, 총 소스={total_sources}, 반환 소스={len(sources)}")
        
        return {
            "sources": sources,
            "total": total_sources,
            "page": page,
            "size": size,
            "query_time_ms": query_time
        }
        
    except Exception as e:
        logger.exception(f"[{request_id}] 문서 소스 목록 조회 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"문서 소스 목록 조회 중 오류 발생: {str(e)}")


@router.get("/documents/source/{source_url:path}", tags=["문서 API"])
async def get_document_by_source(
    source_url: str = Path(..., description="문서 소스 URL (부분 일치 검색 가능)"),
    exact_match: bool = Query(False, description="정확히 일치하는 URL만 검색할지 여부")
):
    """소스 URL 패턴과 일치하는 문서 내용을 가져옵니다.
    
    Args:
        source_url: 문서 소스 URL 패턴
        exact_match: 정확히 일치하는 URL만 검색할지 여부
    
    Returns:
        일치하는 문서 목록
    """
    
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] 문서 내용 요청: source_url={source_url}, exact_match={exact_match}")
    
    try:
        # OpenSearch 클라이언트 가져오기
        from services.opensearch import get_opensearch_client
        client = get_opensearch_client()
        
        # 인덱스가 없으면 오류
        if not client.indices.exists(index=INDEX_NAME):
            logger.error(f"[{request_id}] 인덱스 '{INDEX_NAME}'가 없습니다")
            raise HTTPException(status_code=404, detail="인덱스가 없습니다")
        
        # 쿼리 구성
        if exact_match:
            # 정확히 일치하는 경우 (기존 코드)
            query_clause = {
                "term": {
                    "metadata.source_url.keyword": source_url
                }
            }
        else:
            # 부분 일치 검색을 위한 wildcard 쿼리
            # 사용자가 입력한 URL 주변에 와일드카드(*) 추가
            # wildcard는 성능이 느릴 수 있음에 주의
            query_clause = {
                "wildcard": {
                    "metadata.source_url.keyword": f"*{source_url}*"
                }
            }
            
            # 또는 regexp 쿼리를 사용할 수도 있음
            # query_clause = {
            #     "regexp": {
            #         "metadata.source_url.keyword": f".*{source_url}.*"
            #     }
            # }
        
        # 전체 쿼리 구성
        search_query = {
            "query": query_clause,
            "size": 1000,  # 최대 청크 수
            "_source": ["text", "metadata"]
        }
        
        # 쿼리 실행
        start_time = time.time()
        response = client.search(
            index=INDEX_NAME,
            body=search_query
        )
        query_time = round((time.time() - start_time) * 1000, 2)
        
        # 결과 처리
        hits = response["hits"]["hits"]
        total_hits = response["hits"]["total"]["value"]
        
        if total_hits == 0:
            logger.error(f"[{request_id}] 문서를 찾을 수 없음: source_url={source_url}")
            raise HTTPException(status_code=404, detail=f"URL 패턴 '{source_url}'에 해당하는 문서를 찾을 수 없습니다")
        
        # 소스 URL로 문서 그룹화
        documents_by_url = {}
        
        for hit in hits:
            source = hit["_source"]
            metadata = source.get("metadata", {})
            url = metadata.get("source_url", "")
            
            if url not in documents_by_url:
                documents_by_url[url] = {
                    "source_url": url,
                    "category": metadata.get("category", ""),
                    "created_at": metadata.get("created_at", ""),
                    "chunks": [],
                    "full_text": ""
                }
            
            documents_by_url[url]["chunks"].append({
                "id": hit["_id"],
                "chunk_id": metadata.get("chunk_id", 0),
                "text": source.get("text", "")
            })
        
        # 각 문서의 청크를 정렬하고 전체 텍스트 구성
        documents = []
        for url, doc in documents_by_url.items():
            # 청크를 chunk_id 순으로 정렬
            doc["chunks"].sort(key=lambda x: x["chunk_id"])
            
            # 전체 텍스트 구성
            doc["full_text"] = "\n\n".join([chunk["text"] for chunk in doc["chunks"]])
            
            # 청크 수 추가
            doc["chunk_count"] = len(doc["chunks"])
            
            documents.append(doc)
        
        logger.info(f"[{request_id}] 문서 내용 완료: 소요시간={query_time}ms, 문서 수={len(documents)}")
        
        # 단일 문서만 요청한 경우 (정확히 일치하는 경우)
        if exact_match and len(documents) == 1:
            return documents[0]
        
        # 여러 문서를 반환하는 경우 (부분 일치 검색)
        return {
            "documents": documents,
            "total": len(documents),
            "query_time_ms": query_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] 문서 내용 조회 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"문서 내용 조회 중 오류 발생: {str(e)}")


@router.get("/categories", tags=["문서 API"])
async def list_categories():
    """모든 카테고리 목록을 가져옵니다.
    
    Returns:
        카테고리 목록
    """
    
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] 카테고리 목록 요청")
    
    try:
        # OpenSearch 클라이언트 가져오기
        from services.opensearch import get_opensearch_client
        client = get_opensearch_client()
        
        # 인덱스가 없으면 오류
        if not client.indices.exists(index=INDEX_NAME):
            logger.error(f"[{request_id}] 인덱스 '{INDEX_NAME}'가 없습니다")
            return {"categories": []}
        
        # 쿼리 구성
        aggs_query = {
            "size": 0,
            "aggs": {
                "categories": {
                    "terms": {
                        "field": "metadata.category.keyword",
                        "size": 1000
                    },
                    "aggs": {
                        "doc_count": {
                            "cardinality": {
                                "field": "metadata.source_url.keyword"
                            }
                        }
                    }
                }
            }
        }
        
        # 쿼리 실행
        start_time = time.time()
        response = client.search(
            index=INDEX_NAME,
            body=aggs_query
        )
        query_time = round((time.time() - start_time) * 1000, 2)
        
        # 결과 처리
        buckets = response["aggregations"]["categories"]["buckets"]
        
        categories = []
        for bucket in buckets:
            category = bucket["key"]
            doc_count = bucket["doc_count"]["value"]
            
            categories.append({
                "name": category,
                "document_count": doc_count
            })
        
        logger.info(f"[{request_id}] 카테고리 목록 완료: 소요시간={query_time}ms, 카테고리 수={len(categories)}")
        
        return {
            "categories": categories,
            "query_time_ms": query_time
        }
        
    except Exception as e:
        logger.exception(f"[{request_id}] 카테고리 목록 조회 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"카테고리 목록 조회 중 오류 발생: {str(e)}")

@router.post("/search/vector", tags=["문서 API"])
async def vector_search(
    text: str = Form(..., description="검색할 텍스트"),
    category: Optional[str] = Form(None, description="카테고리 필터"),
    top_k: int = Form(10, ge=1, le=100, description="반환할 결과 수")
):
    """벡터 검색을 수행합니다.
    
    Args:
        text: 검색할 텍스트
        category: 카테고리 필터
        top_k: 반환할 결과 수
    
    Returns:
        유사한 문서 목록
    """
    
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] 벡터 검색 요청: 텍스트='{text[:50]}...', top_k={top_k}")
    
    try:
        # OpenSearch 클라이언트 가져오기
        from services.opensearch import get_opensearch_client
        client = get_opensearch_client()
        
        # 인덱스가 없으면 오류
        if not client.indices.exists(index=INDEX_NAME):
            logger.error(f"[{request_id}] 인덱스 '{INDEX_NAME}'가 없습니다")
            return {"results": [], "total": 0}
        
        # 텍스트 임베딩 생성
        from services.document_processor import embeddings
        import asyncio
        vector_embedding = await asyncio.to_thread(embeddings.embed_query, text)
        
        # 쿼리 구성
        search_query = {
            "size": top_k,
            "query": {
                "knn": {
                    "vector_field": {
                        "vector": vector_embedding,
                        "k": top_k
                    }
                }
            }
        }
        
        # 카테고리 필터 추가
        if category:
            search_query["post_filter"] = {
                "term": {
                    "metadata.category.keyword": category
                }
            }
        
        # 쿼리 실행
        start_time = time.time()
        response = client.search(
            index=INDEX_NAME,
            body=search_query
        )
        search_time = round((time.time() - start_time) * 1000, 2)
        
        # 결과 처리
        hits = response["hits"]["hits"]
        
        results = []
        for hit in hits:
            source = hit["_source"]
            metadata = source.get("metadata", {})
            
            result_item = {
                "id": hit["_id"],
                "score": hit.get("_score", 0),
                "text": source.get("text", ""),
                "source_url": metadata.get("source_url", ""),
                "category": metadata.get("category", ""),
                "chunk_id": metadata.get("chunk_id", 0),
                "created_at": metadata.get("created_at", "")
            }
            
            results.append(result_item)
        
        logger.info(f"[{request_id}] 벡터 검색 완료: 소요시간={search_time}ms, 결과 수={len(results)}")
        
        return {
            "results": results,
            "total": len(results),
            "search_time_ms": search_time
        }
        
    except Exception as e:
        logger.exception(f"[{request_id}] 벡터 검색 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"벡터 검색 중 오류 발생: {str(e)}")

@router.get("/documents/{document_id}/exists", tags=["문서 API"])
async def check_document_exists(
    document_id: str = Path(..., description="확인할 문서 ID")
):
    """문서 존재 여부를 확인합니다.
    
    Args:
        document_id: 확인할 문서 ID
    
    Returns:
        dict: 존재 여부와 문서 정보
    """
    
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] 문서 존재 확인 요청: document_id={document_id}")
    
    try:
        # OpenSearch 클라이언트 가져오기
        from services.opensearch import get_opensearch_client
        from opensearchpy.exceptions import NotFoundError
        client = get_opensearch_client()
        
        # 인덱스가 없으면 오류
        if not client.indices.exists(index=INDEX_NAME):
            logger.error(f"[{request_id}] 인덱스 '{INDEX_NAME}'가 없습니다")
            return {"exists": False, "document_id": document_id}
        
        try:
            # 문서 조회
            doc_response = client.get(index=INDEX_NAME, id=document_id)
            doc_source = doc_response["_source"]
            metadata = doc_source.get("metadata", {})
            
            logger.info(f"[{request_id}] 문서 존재 확인 완료: document_id={document_id}, 존재함")
            
            return {
                "exists": True,
                "document_id": document_id,
                "source_url": metadata.get("source_url"),
                "category": metadata.get("category"),
                "created_at": metadata.get("created_at"),
                "chunk_id": metadata.get("chunk_id")
            }
        except NotFoundError:
            # OpenSearch의 NotFoundError를 명시적으로 처리
            logger.info(f"[{request_id}] 문서 존재 확인 완료: document_id={document_id}, 존재하지 않음")
            return {
                "exists": False,
                "document_id": document_id
            }            
        except Exception as e:
            if "not_found" in str(e).lower():
                logger.info(f"[{request_id}] 문서 존재 확인 완료: document_id={document_id}, 존재하지 않음")
                return {
                    "exists": False,
                    "document_id": document_id
                }
            else:
                raise
        
    except Exception as e:
        logger.exception(f"[{request_id}] 문서 존재 확인 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"문서 존재 확인 중 오류 발생: {str(e)}")
    

@router.delete("/documents/{document_id}", tags=["문서 API"], response_model=DeleteResponse)
async def delete_document_by_id(
    document_id: str = Path(..., description="삭제할 문서 ID")
):
    """문서 ID로 특정 문서를 삭제합니다.
    
    Args:
        document_id: 삭제할 문서의 고유 ID
    
    Returns:
        DeleteResponse: 삭제 결과 응답
    
    Raises:
        HTTPException: 문서를 찾을 수 없거나 삭제 실패 시
    """
    
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] 문서 삭제 요청: document_id={document_id}")
    
    try:
        # OpenSearch 클라이언트 가져오기
        from services.opensearch import get_opensearch_client
        from opensearchpy.exceptions import NotFoundError
        client = get_opensearch_client()
        
        # 인덱스가 없으면 오류
        if not client.indices.exists(index=INDEX_NAME):
            logger.error(f"[{request_id}] 인덱스 '{INDEX_NAME}'가 없습니다")
            raise HTTPException(status_code=404, detail="인덱스가 없습니다")
        
        # 문서 존재 여부 확인
        try:
            doc_response = client.get(index=INDEX_NAME, id=document_id)
            doc_source = doc_response["_source"]
            metadata = doc_source.get("metadata", {})
            source_url = metadata.get("source_url", "Unknown")
            
            logger.info(f"[{request_id}] 삭제할 문서 발견: source_url={source_url}")
        except NotFoundError:
            logger.error(f"[{request_id}] 문서를 찾을 수 없음: document_id={document_id}")
            raise HTTPException(
                status_code=404,
                detail=f"문서 ID '{document_id}'를 찾을 수 없습니다."
            )            
        except Exception as e:
            if "not_found" in str(e).lower():
                logger.error(f"[{request_id}] 문서를 찾을 수 없음: document_id={document_id}")
                raise HTTPException(
                    status_code=404,
                    detail=f"문서 ID '{document_id}'를 찾을 수 없습니다."
                )
            else:
                raise
        
        # 문서 삭제 실행
        start_time = time.time()
        delete_response = client.delete(
            index=INDEX_NAME,
            id=document_id
        )
        delete_time = round((time.time() - start_time) * 1000, 2)
        
        # 삭제 결과 확인
        if delete_response.get("result") == "deleted":
            logger.info(f"[{request_id}] 문서 삭제 완료: document_id={document_id}, 소요시간={delete_time}ms")
            
            return DeleteResponse(
                success=True,
                message=f"문서 ID '{document_id}'가 성공적으로 삭제되었습니다.",
                deleted_count=1,
                deleted_ids=[document_id]
            )
        else:
            logger.error(f"[{request_id}] 문서 삭제 실패: {delete_response}")
            raise HTTPException(
                status_code=500,
                detail=f"문서 삭제에 실패했습니다: {delete_response.get('result', 'Unknown')}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] 문서 삭제 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"문서 삭제 중 오류 발생: {str(e)}")


@router.delete("/documents/batch", tags=["문서 API"], response_model=DeleteResponse)
async def delete_documents_batch(
    request: BatchDeleteRequest
):
    """여러 문서를 일괄 삭제합니다.
    
    Args:
        request: 삭제할 문서 ID 목록
    
    Returns:
        DeleteResponse: 일괄 삭제 결과
    """
    
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] 일괄 문서 삭제 요청: 문서 수={len(request.document_ids)}")
    
    try:
        # OpenSearch 클라이언트 가져오기
        from services.opensearch import get_opensearch_client
        client = get_opensearch_client()
        
        # 인덱스가 없으면 오류
        if not client.indices.exists(index=INDEX_NAME):
            logger.error(f"[{request_id}] 인덱스 '{INDEX_NAME}'가 없습니다")
            raise HTTPException(status_code=404, detail="인덱스가 없습니다")
        
        # 일괄 삭제 요청 구성
        bulk_body = []
        for doc_id in request.document_ids:
            bulk_body.append({
                "delete": {
                    "_index": INDEX_NAME,
                    "_id": doc_id
                }
            })
        
        # 일괄 삭제 실행
        start_time = time.time()
        bulk_response = client.bulk(body=bulk_body)
        delete_time = round((time.time() - start_time) * 1000, 2)
        
        # 결과 처리
        deleted_ids = []
        failed_ids = []
        
        for item in bulk_response["items"]:
            if "delete" in item:
                delete_result = item["delete"]
                doc_id = delete_result["_id"]
                
                if delete_result.get("result") == "deleted":
                    deleted_ids.append(doc_id)
                else:
                    failed_ids.append(doc_id)
                    logger.warning(f"[{request_id}] 문서 삭제 실패: {doc_id} - {delete_result}")
        
        success_count = len(deleted_ids)
        fail_count = len(failed_ids)
        
        logger.info(f"[{request_id}] 일괄 삭제 완료: 성공={success_count}, 실패={fail_count}, 소요시간={delete_time}ms")
        
        message = f"총 {len(request.document_ids)}개 문서 중 {success_count}개 삭제 완료"
        if fail_count > 0:
            message += f", {fail_count}개 실패"
        
        return DeleteResponse(
            success=success_count > 0,
            message=message,
            deleted_count=success_count,
            deleted_ids=deleted_ids
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] 일괄 문서 삭제 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"일괄 문서 삭제 중 오류 발생: {str(e)}")