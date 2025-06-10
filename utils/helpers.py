import time
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import re
import hashlib
from urllib.parse import urlparse, unquote

from logger import logger
from config import OPENSEARCH_CONFIG

INDEX_NAME = OPENSEARCH_CONFIG["index_name"]

def generate_request_id() -> str:
    """고유한 요청 ID를 생성합니다."""
    return f"req_{int(time.time() * 1000)}"

def sanitize_text(text: str) -> str:
    """텍스트를 정제합니다."""
    # HTML 태그 제거
    text = re.sub(r'<[^>]*>', '', text)
    # 여러 개의 공백을 하나로 줄임
    text = re.sub(r'\s+', ' ', text)
    # 여러 개의 줄바꿈을 하나로 줄임
    text = re.sub(r'\n+', '\n', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text

def truncate_text(text: str, max_length: int = 200) -> str:
    """텍스트를 지정된 길이로 자릅니다."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def normalize_url(url: str) -> str:
    """URL을 정규화합니다."""
    # URL 디코딩
    url = unquote(url)
    
    # 파싱
    parsed = urlparse(url)
    
    # 스키마 및 호스트 소문자로 변환
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    
    # 경로에서 마지막 슬래시 제거 (있는 경우)
    path = parsed.path
    if path.endswith('/') and len(path) > 1:
        path = path[:-1]
    
    # 정규화된 URL 재구성
    normalized = f"{scheme}://{netloc}{path}"
    
    # 쿼리 파라미터가 있는 경우 추가
    if parsed.query:
        normalized += f"?{parsed.query}"
    
    # 프래그먼트가 있는 경우 추가
    if parsed.fragment:
        normalized += f"#{parsed.fragment}"
    
    return normalized

def generate_document_id(url: str) -> str:
    """URL에서 문서 ID를 생성합니다."""
    # URL 정규화
    normalized_url = normalize_url(url)
    
    # MD5 해시 생성
    return hashlib.md5(normalized_url.encode()).hexdigest()

def format_iso_date(dt: Optional[datetime] = None) -> str:
    """ISO 8601 형식의 날짜 문자열을 반환합니다."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.isoformat()

def parse_iso_date(date_str: str) -> Optional[datetime]:
    """ISO 8601 형식의 날짜 문자열을 파싱합니다."""
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None

async def check_index_exists(client: Any, index_name: str = INDEX_NAME) -> bool:
    """인덱스가 존재하는지 확인합니다."""
    try:
        return client.indices.exists(index=index_name)
    except Exception as e:
        logger.error(f"인덱스 확인 중 오류: {e}")
        return False

def format_search_results(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """검색 결과를 형식화합니다."""
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
    
    return results

def combine_search_results(vector_hits: List[Dict[str, Any]], text_hits: List[Dict[str, Any]], vector_weight: float = 0.5) -> List[Dict[str, Any]]:
    """벡터 검색과 텍스트 검색 결과를 병합합니다."""
    # 문서 ID로 맵 생성
    combined_docs = {}
    
    # 벡터 결과 추가
    for hit in vector_hits:
        doc_id = hit["_id"]
        combined_docs[doc_id] = {
            "id": doc_id,
            "source": hit["_source"],
            "vector_score": hit["_score"],
            "text_score": 0.0,
            "combined_score": hit["_score"] * vector_weight
        }
    
    # 텍스트 결과 추가 또는 병합
    text_weight = 1.0 - vector_weight
    for hit in text_hits:
        doc_id = hit["_id"]
        if doc_id in combined_docs:
            # 이미 벡터 결과에 있는 경우 텍스트 스코어 추가
            combined_docs[doc_id]["text_score"] = hit["_score"]
            combined_docs[doc_id]["combined_score"] += hit["_score"] * text_weight
        else:
            # 벡터 결과에 없는 경우 새로 추가
            combined_docs[doc_id] = {
                "id": doc_id,
                "source": hit["_source"],
                "vector_score": 0.0,
                "text_score": hit["_score"],
                "combined_score": hit["_score"] * text_weight
            }
    
    # 통합 스코어로 정렬
    sorted_docs = sorted(
        combined_docs.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )
    
    return sorted_docs

def create_vector_query(vector_embedding: List[float], category: Optional[str] = None, size: int = 10) -> Dict[str, Any]:
    """벡터 검색 쿼리를 생성합니다."""
    query = {
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": "vector_field",
                        "query_value": vector_embedding,
                        "space_type": "cosinesimil"
                    }
                }
            }
        }
    }
    
    # 카테고리 필터 추가
    if category:
        query["query"] = {
            "bool": {
                "must": [query["query"]],
                "filter": [{"term": {"metadata.category.keyword": category}}]
            }
        }
    
    return query

def create_text_query(text: str, category: Optional[str] = None, size: int = 10) -> Dict[str, Any]:
    """텍스트 검색 쿼리를 생성합니다."""
    query = {
        "size": size,
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "text": text
                        }
                    }
                ]
            }
        }
    }
    
    # 카테고리 필터 추가
    if category:
        query["query"]["bool"]["filter"] = [
            {"term": {"metadata.category.keyword": category}}
        ]
    
    return query

def create_hybrid_query(text: str, vector_embedding: List[float], category: Optional[str] = None, size: int = 10) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """하이브리드 검색을 위한 벡터 및 텍스트 쿼리를 생성합니다."""
    # 더 많은 결과를 가져와서 후처리
    vector_query = create_vector_query(vector_embedding, category, size * 3)
    text_query = create_text_query(text, category, size * 3)
    
    return vector_query, text_query

def get_opensearch_mapping() -> Dict[str, Any]:
    """OpenSearch 인덱스 매핑을 반환합니다."""
    return {
        "mappings": {
            "properties": {
                "category": {
                    "type": "keyword"
                },
                "content": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword"
                        }
                    },
                    "analyzer": "korean"
                },
                "created_at": {
                    "type": "date"
                },
                "metadata": {
                    "properties": {
                        "category": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "chunk_id": {
                            "type": "long"
                        },
                        "created_at": {
                            "type": "date"
                        },
                        "source_url": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        }
                    }
                },
                "source_url": {
                    "type": "keyword"
                },
                "text": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": 768,
                    "method": {
                        "engine": "nmslib",
                        "space_type": "cosinesimil",
                        "name": "hnsw",
                        "parameters": {}
                    }
                }
            }
        }
    }