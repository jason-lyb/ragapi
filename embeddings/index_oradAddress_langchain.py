"""
최종 3인덱스 고속 병렬 RAG 시스템
Road + Jibun + Building (지역포함, aliases제거)
"""

import os
import json
import time
import re
import psycopg2
import psycopg2.extras
from datetime import datetime
from typing import List, Dict, Any, Generator, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

# LangChain 라이브러리
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.schema import Document

# OpenSearch 연결
import requests
from requests.auth import HTTPBasicAuth

# 설정 (config 파일에서 로드)
from config import (
    OPENSEARCH_CONFIG4, 
    POSTGRESQL_CONFIG,
    SEARCH_CONFIG,
    INDEX_CONFIG
)

class FinalThreeIndexRAG:
    """최종 3인덱스 RAG 시스템 - Road, Jibun, Building"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """최종 3인덱스 RAG 시스템 초기화"""
        
        # 설정 로드 (기존 config 파일 참조)
        opensearch_config = config.get('opensearch', OPENSEARCH_CONFIG4) if config else OPENSEARCH_CONFIG4
        self.postgresql_config = config.get('postgresql', POSTGRESQL_CONFIG) if config else POSTGRESQL_CONFIG
        
        # OpenSearch 연결 설정
        self.opensearch_url = opensearch_config.get('url')
        self.base_index_name = opensearch_config.get('index_name', 'ko-address')
        self.username = opensearch_config.get('username')
        self.password = opensearch_config.get('password')
        self.verify_ssl = opensearch_config.get('verify_ssl', False)
        self.auth = HTTPBasicAuth(self.username, self.password)
        self.headers = {'Content-Type': 'application/json'}
        
        # 3개 인덱스 설정
        self.indexes = {
            'road': f"{self.base_index_name}-road",         # 도로명주소 전용
            'jibun': f"{self.base_index_name}-jibun",       # 지번주소 전용
            'building': f"{self.base_index_name}-building"   # 빌딩명+지역 전용
        }
        
        # 한국어 SRoBERTa 임베딩 모델 초기화
        print("🤖 한국어 SRoBERTa 모델 로딩 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sroberta-multitask',
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 16
            }
        )
        print("✅ 한국어 SRoBERTa 모델 로딩 완료")
        
        # 벡터 스토어들
        self.vector_stores = {}
        
        # PostgreSQL 연결
        self.pg_connection = None
        
        # 통계
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None
        }

    def connect_postgresql(self) -> bool:
        """PostgreSQL 연결"""
        try:
            self.pg_connection = psycopg2.connect(
                host=self.postgresql_config.get('host'),
                port=self.postgresql_config.get('port'),
                database=self.postgresql_config.get('database'),
                user=self.postgresql_config.get('user'),
                password=self.postgresql_config.get('password'),
                sslmode=self.postgresql_config.get('sslmode', 'disable')
            )
            self.pg_connection.autocommit = False
            print("✅ PostgreSQL 연결 성공")
            return True
        except Exception as e:
            print(f"❌ PostgreSQL 연결 실패: {e}")
            return False

    def check_opensearch_connection(self) -> bool:
        """OpenSearch 연결 확인"""
        try:
            response = requests.get(
                self.opensearch_url,
                auth=self.auth,
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=120
            )
            print(f"✅ OpenSearch 연결 확인: {response.status_code}")
            return response.status_code < 400
        except Exception as e:
            print(f"❌ OpenSearch 연결 실패: {e}")
            return False

    def create_three_indexes(self) -> bool:
        """3개 인덱스 생성 - Road, Jibun, Building"""
        
        print("🔧 3인덱스 생성 시작 (Road + Jibun + Building)")
        
        # OpenSearch 연결 재확인
        if not self.check_opensearch_connection():
            print("❌ OpenSearch 연결 실패 - 인덱스 생성 중단")
            return False
        
        # 기존 인덱스들 삭제
        for index_type, index_name in self.indexes.items():
            self._delete_index_if_exists(index_name)
        
        # 공통 인덱스 설정
        base_config = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "30s",
                    "knn": True,
                    "max_ngram_diff": 5,
                    "analysis": {
                        "analyzer": {
                            "korean_analyzer": {
                                "type": "custom",
                                "tokenizer": "nori_tokenizer",
                                "filter": ["lowercase", "nori_part_of_speech"]
                            },
                            "address_analyzer": {
                                "type": "custom",
                                "tokenizer": "keyword",
                                "filter": ["lowercase", "trim"]
                            },
                            "address_ngram_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "address_ngram_filter"]
                            },
                            "address_search_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase"]
                            }
                        },
                        "filter": {
                            "address_ngram_filter": {
                                "type": "ngram",
                                "min_gram": 2,
                                "max_gram": 4,
                                "token_chars": ["letter", "digit"]
                            }
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "korean_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "address": {"type": "text", "analyzer": "address_analyzer"}
                        }
                    },
                    "address_field": {
                        "type": "text",
                        "analyzer": "address_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "ngram": {
                                "type": "text",
                                "analyzer": "address_ngram_analyzer",
                                "search_analyzer": "address_search_analyzer"
                            },
                            "partial": {
                                "type": "text",
                                "analyzer": "korean_analyzer"
                            }
                        }
                    },
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16
                            }
                        }
                    },
                    "postal_code": {"type": "keyword"},
                    "region": {"type": "keyword"},
                    "city": {"type": "keyword"},
                    "district": {"type": "keyword"},
                    "building_name": {"type": "keyword"},
                    "search_type": {"type": "keyword"},
                    "created_at": {"type": "date"}
                }
            }
        }
        
        # 각 인덱스 생성
        created_indexes = []
        
        for index_type, index_name in self.indexes.items():
            try:
                print(f"🔨 {index_type} 인덱스 생성 중: {index_name}")
                
                response = requests.put(
                    f"{self.opensearch_url}/{index_name}",
                    headers=self.headers,
                    data=json.dumps(base_config),
                    auth=self.auth,
                    verify=self.verify_ssl,
                    timeout=120
                )
                
                if response.status_code in [200, 201]:
                    print(f"  ✅ {index_type} 인덱스 생성 성공: {index_name}")
                    created_indexes.append(index_name)
                    time.sleep(2)
                else:
                    print(f"  ❌ {index_type} 인덱스 생성 실패: {response.status_code}")
                    print(f"  📄 응답: {response.text[:500]}")
                    
                    # 이미 생성된 인덱스들 정리
                    for created_index in created_indexes:
                        self._delete_index_if_exists(created_index)
                    return False
                    
            except Exception as e:
                print(f"  ❌ {index_type} 인덱스 생성 오류: {e}")
                for created_index in created_indexes:
                    self._delete_index_if_exists(created_index)
                return False
        
        # 인덱스 안정화 대기
        print("⏳ 인덱스 안정화 대기 중...")
        time.sleep(10)
        
        # 최종 확인
        all_created = True
        for index_type, index_name in self.indexes.items():
            try:
                response = requests.head(
                    f"{self.opensearch_url}/{index_name}",
                    auth=self.auth,
                    verify=self.verify_ssl,
                    timeout=120
                )
                
                if response.status_code == 200:
                    print(f"  ✅ {index_type} 인덱스 최종 확인 완료: {index_name}")
                else:
                    print(f"  ❌ {index_type} 인덱스 최종 확인 실패: {response.status_code}")
                    all_created = False
                    
            except Exception as e:
                print(f"  ❌ {index_type} 인덱스 확인 오류: {e}")
                all_created = False
        
        if all_created:
            print("🎉 모든 3인덱스 생성 완료!")
            print(f"📂 생성된 인덱스: {len(self.indexes)}개")
            for index_type, index_name in self.indexes.items():
                print(f"  • {index_type.upper()}: {index_name}")
            return True
        else:
            print("❌ 일부 인덱스 생성 실패")
            return False

    def _delete_index_if_exists(self, index_name: str):
        """인덱스 삭제"""
        try:
            response = requests.head(
                f"{self.opensearch_url}/{index_name}",
                auth=self.auth,
                verify=self.verify_ssl
            )
            
            if response.status_code == 200:
                print(f"기존 인덱스 {index_name} 삭제 중...")
                delete_response = requests.delete(
                    f"{self.opensearch_url}/{index_name}",
                    auth=self.auth,
                    verify=self.verify_ssl,
                    timeout=120
                )
                
                if delete_response.status_code in [200, 404]:
                    print(f"기존 인덱스 {index_name} 삭제 완료")
                    time.sleep(2)
                    
        except Exception as e:
            print(f"인덱스 삭제 중 오류 (무시): {e}")

    def extract_address_components(self, address: str) -> Dict[str, str]:
        """주소에서 핵심 구성요소 추출"""
        components = {
            "region": "",      # 시/도
            "city": "",        # 시/군/구  
            "district": "",    # 동/읍/면
            "road": "",        # 도로명
            "dong": "",        # 동명 (지번용)
            "building": ""     # 건물명
        }
        
        if not address:
            return components
        
        # 괄호 안의 내용 추출 (동명, 건물명)
        parenthesis_match = re.search(r'\(([^)]+)\)', address)
        if parenthesis_match:
            parenthesis_content = parenthesis_match.group(1)
            if ',' in parenthesis_content:
                parts = parenthesis_content.split(',')
                components["dong"] = parts[0].strip()
                components["building"] = parts[1].strip() if len(parts) > 1 else ""
            else:
                if '동' in parenthesis_content or '읍' in parenthesis_content or '면' in parenthesis_content:
                    components["dong"] = parenthesis_content.strip()
                else:
                    components["building"] = parenthesis_content.strip()
        
        # 괄호 제거하고 기본 주소 파싱
        clean_address = re.sub(r'\([^)]*\)', '', address).strip()
        parts = clean_address.split()
        
        if len(parts) >= 1:
            components["region"] = parts[0]  # 서울특별시, 경기도 등
        if len(parts) >= 2:
            components["city"] = parts[1]    # 강남구, 성남시 등
        if len(parts) >= 3:
            # 도로명 찾기
            for i, part in enumerate(parts[2:], 2):
                if '로' in part or '길' in part or '대로' in part:
                    components["road"] = part
                    break
        
        return components

    def get_sido_short(self, sido: str) -> str:
        """시도명 단축형 변환"""
        sido_mapping = {
            '서울특별시': '서울',
            '부산광역시': '부산', 
            '대구광역시': '대구',
            '인천광역시': '인천',
            '광주광역시': '광주',
            '대전광역시': '대전',
            '울산광역시': '울산',
            '세종특별자치시': '세종',
            '경기도': '경기',
            '강원도': '강원',
            '충청북도': '충북',
            '충청남도': '충남',
            '전라북도': '전북',
            '전라남도': '전남',
            '경상북도': '경북',
            '경상남도': '경남',
            '제주특별자치도': '제주'
        }
        return sido_mapping.get(sido, sido)

    def get_sigungu_short(self, sigungu: str) -> str:
        """시군구명 단축형 변환"""
        if sigungu.endswith('구'):
            return sigungu[:-1]  # 강남구 → 강남
        elif sigungu.endswith('시'):
            return sigungu[:-1]  # 수원시 → 수원
        elif sigungu.endswith('군'):
            return sigungu[:-1]  # 양평군 → 양평
        return sigungu

    def get_dong_short(self, dong: str) -> str:
        """동명 단축형 변환"""
        if dong.endswith('동'):
            return dong[:-1]  # 역삼동 → 역삼
        elif dong.endswith('읍'):
            return dong[:-1]  # 신도읍 → 신도
        elif dong.endswith('면'):
            return dong[:-1]  # 외동면 → 외동
        return dong

    def extract_building_number(self, address: str) -> str:
        """주소에서 건물번호 정확히 추출"""
        if not address:
            return ""
        
        # 도로명 뒤의 첫 번째 숫자 패턴
        road_number_pattern = r'(?:로|길|대로)\s+(\d+(?:-\d+)?)'
        match = re.search(road_number_pattern, address)
        
        if match:
            return match.group(1)
        
        # 일반적인 숫자 패턴 (마지막 숫자)
        number_patterns = re.findall(r'\d+(?:-\d+)?', address)
        if number_patterns:
            return number_patterns[-1]
        
        return ""

    def create_road_content(self, road_address: str) -> str:
        """도로명주소 콘텐츠 생성 (빌딩명 제외)"""
        if not road_address:
            return ""
        
        # 괄호 안 내용 제거 (동명, 빌딩명 모두 제거)
        clean_address = re.sub(r'\([^)]*\)', '', road_address).strip()
        
        # 주소 구성요소 추출
        components = self.extract_address_components(clean_address)
        
        # 순수 주소 정보만 조합
        road_parts = []
        
        if components.get('region'):
            road_parts.append(components['region'])
        
        if components.get('city'):
            road_parts.append(components['city'])
        
        if components.get('road'):
            road_parts.append(components['road'])
        
        # 건물번호
        building_number = self.extract_building_number(clean_address)
        if building_number:
            road_parts.append(building_number)
        
        return " ".join(road_parts)

    def create_jibun_content(self, jibun_address: str) -> str:
        """지번주소 콘텐츠 생성"""
        if not jibun_address:
            return ""
        
        # 괄호 제거하고 순수 지번주소만
        clean_address = re.sub(r'\([^)]*\)', '', jibun_address).strip()
        return clean_address

    def create_building_content(self, building_name: str, road_address: str) -> str:
        """빌딩 콘텐츠 생성 (지역 포함, aliases 제거)"""
        if not building_name:
            return ""
        
        content_parts = [building_name]
        
        # 지역 정보 추출
        components = self.extract_address_components(road_address)
        
        # 시도 + 빌딩명
        sido_short = self.get_sido_short(components.get('region', ''))
        if sido_short:
            content_parts.append(f"{sido_short} {building_name}")
        
        # 시군구 + 빌딩명
        sigungu = components.get('city', '')
        if sigungu:
            content_parts.append(f"{sigungu} {building_name}")
            
            sigungu_short = self.get_sigungu_short(sigungu)
            if sigungu_short != sigungu:
                content_parts.append(f"{sigungu_short} {building_name}")
        
        # 동 + 빌딩명
        dong = components.get('dong', '')
        if dong:
            content_parts.append(f"{dong} {building_name}")
            
            dong_short = self.get_dong_short(dong)
            if dong_short != dong:
                content_parts.append(f"{dong_short} {building_name}")
        
        return " ".join(content_parts)

    def get_address_count(self) -> int:
        """주소 데이터 총 개수 조회"""
        try:
            cursor = self.pg_connection.cursor()
            
            count_query = """
            SELECT COUNT(*) 
            FROM address.building_info
            WHERE road_name IS NOT NULL 
              AND building_main_number IS NOT NULL
            """
            
            cursor.execute(count_query)
            total_count = cursor.fetchone()[0]
            cursor.close()
            self.pg_connection.commit()
            
            print(f"📊 총 처리 대상: {total_count:,}개 주소")
            return total_count
            
        except Exception as e:
            print(f"❌ 카운트 조회 오류: {e}")
            if self.pg_connection:
                self.pg_connection.rollback()
            return 0

    def stream_address_documents(self, batch_size: int = 1000) -> Generator[Dict[str, List[Document]], None, None]:
        """PostgreSQL에서 주소 데이터를 스트리밍으로 가져와 3인덱스 Document 생성"""
        
        try:
            cursor = self.pg_connection.cursor(name='address_cursor')
            
            query = """
            SELECT DISTINCT 
                road_address, 
                jibun_address, 
                base_zip_code,
                building_name
            FROM (
                SELECT 
                    CASE 
                        WHEN COALESCE(building_sub_number, 0) = 0 THEN 
                            COALESCE(sido_name, '') || ' ' || 
                            COALESCE(road_name, '') || ' ' || 
                            COALESCE(building_main_number::TEXT, '') ||
                            CASE 
                                WHEN COALESCE(eupmyeondong_name, '') != '' THEN 
                                    ' (' || eupmyeondong_name || 
                                        CASE 
                                            WHEN sigungu_building_name IS NOT NULL THEN ', ' || sigungu_building_name 
                                            ELSE '' 
                                        END || ')'
                                ELSE ''
                            END
                        ELSE
                            COALESCE(sido_name, '') || ' ' || 
                            COALESCE(road_name, '') || ' ' || 
                            COALESCE(building_main_number::TEXT, '') || '-' || 
                            COALESCE(building_sub_number::TEXT, '') ||
                            CASE 
                                WHEN COALESCE(eupmyeondong_name, '') != '' THEN 
                                    ' (' || eupmyeondong_name || 
                                        CASE 
                                            WHEN sigungu_building_name IS NOT NULL THEN ', ' || sigungu_building_name 
                                            ELSE '' 
                                        END || ')'
                                ELSE ''
                            END
                    END AS road_address,   
                    CASE 
                        WHEN COALESCE(lot_sub_number, 0) = 0 THEN 
                            COALESCE(sido_name, '') || ' ' || 
                            COALESCE(eupmyeondong_name, '') || ' ' || 
                            COALESCE(legal_ri_name, '') || ' ' || 
                            COALESCE(lot_main_number::TEXT, '')
                        ELSE
                            COALESCE(sido_name, '') || ' ' || 
                            COALESCE(eupmyeondong_name, '') || ' ' || 
                            COALESCE(legal_ri_name, '') || ' ' || 
                            COALESCE(lot_main_number::TEXT, '') || '-' || 
                            COALESCE(lot_sub_number::TEXT, '')
                    END AS jibun_address,
                    base_zip_code,
                    sigungu_building_name as building_name
                FROM address.building_info
                WHERE road_name IS NOT NULL 
                  AND building_main_number IS NOT NULL
            ) addr_data
            ORDER BY road_address
            """
            
            print("🔍 PostgreSQL 쿼리 실행 중...")
            cursor.execute(query)
            
            batch_count = 0
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                
                batch_count += 1
                print(f"📦 배치 {batch_count} 로드: {len(rows)}개 레코드")
                
                # 3인덱스 Document 생성
                document_sets = {"road": [], "jibun": [], "building": []}
                
                for row in rows:
                    road_address = row[0] if row[0] else ""
                    jibun_address = row[1] if row[1] else ""
                    postal_code = row[2] if row[2] else ""
                    building_name = row[3] if row[3] else ""
                    
                    # 주소 구성요소 추출
                    components = self.extract_address_components(road_address)
                    
                    # 공통 메타데이터
                    base_metadata = {
                        "road_address": road_address.strip(),
                        "jibun_address": jibun_address.strip(),
                        "postal_code": postal_code.strip(),
                        "region": components.get('region', ''),
                        "city": components.get('city', ''),
                        "district": components.get('dong', ''),
                        "building_name": building_name.strip(),
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Road Document (도로명주소, 빌딩명 제외)
                    road_content = self.create_road_content(road_address)
                    if road_content.strip():
                        road_doc = Document(
                            page_content=road_content,
                            metadata={**base_metadata, "search_type": "road"}
                        )
                        document_sets["road"].append(road_doc)
                    
                    # Jibun Document (지번주소)
                    if jibun_address and jibun_address != road_address:
                        jibun_content = self.create_jibun_content(jibun_address)
                        if jibun_content.strip():
                            jibun_doc = Document(
                                page_content=jibun_content,
                                metadata={**base_metadata, "search_type": "jibun"}
                            )
                            document_sets["jibun"].append(jibun_doc)
                    
                    # Building Document (빌딩명+지역, aliases 제거)
                    if building_name:
                        building_content = self.create_building_content(building_name, road_address)
                        if building_content.strip():
                            building_doc = Document(
                                page_content=building_content,
                                metadata={**base_metadata, "search_type": "building"}
                            )
                            document_sets["building"].append(building_doc)
                
                yield document_sets
            
            cursor.close()
            self.pg_connection.commit()
            print(f"✅ 총 {batch_count}개 배치 스트리밍 완료")
            
        except Exception as e:
            print(f"❌ 주소 스트리밍 오류: {e}")
            if self.pg_connection:
                self.pg_connection.rollback()

    def initialize_vector_stores(self):
        """벡터 스토어들 초기화"""
        try:
            for index_type, index_name in self.indexes.items():
                self.vector_stores[index_type] = OpenSearchVectorSearch(
                    index_name=index_name,
                    embedding_function=self.embeddings,
                    opensearch_url=self.opensearch_url,
                    http_auth=(self.username, self.password),
                    use_ssl=False,
                    verify_certs=False,
                    ssl_assert_hostname=False,
                    ssl_show_warn=False,
                    timeout=120,
                    max_retries=3
                )
                print(f"✅ {index_type} 벡터 스토어 초기화 완료")
                
        except Exception as e:
            print(f"❌ 벡터 스토어 초기화 실패: {e}")
            raise

    def high_speed_parallel_processing(self,
                                     data_batch_size: int = 3000,
                                     indexing_batch_size: int = 100,
                                     num_workers: int = 6) -> bool:
        """고속 3인덱스 병렬 처리"""

        import gc        
        
        print(f"🚀 고속 3인덱스 병렬 처리 시작")
        print(f"📊 데이터 배치: {data_batch_size}, 인덱싱 배치: {indexing_batch_size}")
        print(f"👷 워커 수: {num_workers} (인덱스당 2개씩)")
        
        # 초기화
        if not self.connect_postgresql():
            return False
        
        total_records = self.get_address_count()
        if total_records == 0:
            return False
        
        # 벡터 스토어 초기화
        if not self.vector_stores:
            self.initialize_vector_stores()
        
        # 통계 및 큐 초기화
        self.stats['start_time'] = time.time()
        self.stats['total_processed'] = 0
        self.stats['successful'] = 0
        self.stats['failed'] = 0
        
        # 인덱스별 작업 큐 (메모리 관리)
        work_queues = {
            'road': queue.Queue(maxsize=5),
            'jibun': queue.Queue(maxsize=5),
            'building': queue.Queue(maxsize=5)
        }
        
        # 결과 큐
        result_queue = queue.Queue()
        
        # 통계 관리용 락
        stats_lock = threading.Lock()
        
        def producer_worker():
            """PostgreSQL에서 데이터를 읽어 각 인덱스별 큐에 분배"""
            try:
                batch_count = 0
                for document_sets in self.stream_address_documents(data_batch_size):
                    batch_count += 1
                    print(f"📦 프로듀서: 배치 {batch_count} 큐에 분배 중...")
                    
                    # 각 인덱스별로 큐에 추가
                    for index_type, documents in document_sets.items():
                        if documents:  # 빈 리스트가 아닌 경우만
                            work_queues[index_type].put(documents)
                            print(f"  📋 {index_type}: {len(documents)}개 큐에 추가")
                
                    # document_sets 메모리 해제
                    del document_sets
                    
                    # 주기적 메모리 정리
                    if batch_count % 5 == 0:
                        gc.collect()

                # 워커 수만큼 종료 신호 추가
                for index_type in work_queues:
                    for _ in range(2):  # 인덱스당 2개 워커
                        work_queues[index_type].put(None)
                
                print(f"✅ 프로듀서 완료: 총 {batch_count}개 배치 분배")
                
            except Exception as e:
                print(f"❌ 프로듀서 오류: {e}")
                # 오류 발생시에도 종료 신호 전송
                for index_type in work_queues:
                    for _ in range(2):
                        work_queues[index_type].put(None)
        
        def consumer_worker(index_type: str, worker_id: int):
            """인덱스별 전용 워커"""
            worker_success = 0
            worker_failed = 0
            worker_name = f"{index_type}-{worker_id}"
            
            print(f"🔧 워커 {worker_name} 시작")
            
            while True:
                try:
                    # 해당 인덱스 큐에서 배치 가져오기
                    documents = work_queues[index_type].get(timeout=120)
                    
                    if documents is None:  # 종료 신호
                        print(f"🏁 워커 {worker_name} 종료")
                        break
                    
                    print(f"  👷 워커 {worker_name}: {len(documents)}개 처리 시작")
                    batch_start_time = time.time()
                    
                    # 소배치로 나누어 인덱싱
                    for i in range(0, len(documents), indexing_batch_size):
                        mini_batch = documents[i:i + indexing_batch_size]
                        
                        # 빠른 인덱싱 (재시도 최소화)
                        for attempt in range(2):  # 최대 2번만 시도
                            try:
                                self.vector_stores[index_type].add_documents(mini_batch)
                                worker_success += len(mini_batch)
                                
                                # Road 인덱스만 전체 통계에 반영
                                if index_type == 'road':
                                    with stats_lock:
                                        self.stats['successful'] += len(mini_batch)
                                        self.stats['total_processed'] += len(mini_batch)
                                
                                break
                            except Exception as e:
                                if attempt == 1:  # 마지막 시도
                                    print(f"    ❌ 워커 {worker_name} 소배치 실패: {str(e)[:50]}...")
                                    worker_failed += len(mini_batch)
                                    
                                    if index_type == 'road':
                                        with stats_lock:
                                            self.stats['failed'] += len(mini_batch)
                                            self.stats['total_processed'] += len(mini_batch)
                                else:
                                    time.sleep(0.3)  # 짧은 재시도 간격
                    
                    batch_time = time.time() - batch_start_time
                    print(f"  ✅ 워커 {worker_name}: {len(documents)}개 완료 ({batch_time:.1f}초)")
                    
                    work_queues[index_type].task_done()
                    
                except queue.Empty:
                    print(f"⏰ 워커 {worker_name} 타임아웃")
                    break
                except Exception as e:
                    print(f"❌ 워커 {worker_name} 오류: {e}")
            
            result_queue.put((worker_name, worker_success, worker_failed))
            return worker_success, worker_failed
        
        # 스레드 시작
        print(f"🚀 3인덱스 병렬 처리 시작 (총 {num_workers + 1}개 워커)")
        
        # 프로듀서 스레드
        producer_thread = threading.Thread(target=producer_worker, daemon=True)
        producer_thread.start()
        
        # 인덱스별 컨슈머 스레드들
        consumer_threads = []
        
        for index_type in ['road', 'jibun', 'building']:
            for worker_id in range(1, 3):  # 인덱스당 2개 워커
                thread = threading.Thread(
                    target=consumer_worker, 
                    args=(index_type, worker_id), 
                    daemon=True
                )
                thread.start()
                consumer_threads.append(thread)
        
        # 진행률 모니터링
        def monitor_progress():
            last_check = 0
            while any(t.is_alive() for t in consumer_threads):
                time.sleep(15)  # 15초마다 체크
                
                # 메모리 사용량 체크
                try:
                    memory_info = psutil.virtual_memory()
                    memory_percent = memory_info.percent
                except:
                    memory_percent = 0

                with stats_lock:
                    current_success = self.stats['successful']
                    current_total = self.stats['total_processed']
                
                if current_success > last_check:
                    elapsed = time.time() - self.stats['start_time']
                    docs_per_min = current_success / (elapsed / 60) if elapsed > 0 else 0
                    progress = current_success / total_records * 100
                    
                    print(f"\n📊 진행률: {progress:.1f}% ({current_success:,}/{total_records:,})")
                    print(f"🚀 속도: {docs_per_min:.0f} docs/min")
                    print(f"🧠 메모리 사용률: {memory_percent:.1f}%")
                    print(f"📂 처리 현황: Road({current_success:,})")
                    
                    # 큐 상태 확인
                    queue_status = []
                    for idx_type, q in work_queues.items():
                        queue_status.append(f"{idx_type}({q.qsize()})")
                    print(f"📋 큐 상태: {', '.join(queue_status)}")
                    
                    if docs_per_min > 0:
                        remaining_docs = total_records - current_success
                        eta_minutes = remaining_docs / docs_per_min
                        print(f"⏱️ 예상 완료: {eta_minutes:.0f}분 후")
                    
                    # 메모리 부족 경고 및 정리
                    if memory_percent > 85:
                        print(f"⚠️ 메모리 부족 경고! 처리 속도가 느려질 수 있습니다.")
                        # 강제 가비지 컬렉션
                        gc.collect()
                        
                    last_check = current_success
        
        # 모니터링 스레드
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        # 모든 스레드 완료 대기
        producer_thread.join()
        print("📦 프로듀서 완료")
        
        for thread in consumer_threads:
            thread.join()
        print("👷 모든 워커 완료")
        
        # 최종 결과 수집
        index_results = {}
        
        while not result_queue.empty():
            try:
                worker_name, success, failed = result_queue.get_nowait()
                index_type = worker_name.split('-')[0]
                
                if index_type not in index_results:
                    index_results[index_type] = {'success': 0, 'failed': 0}
                
                index_results[index_type]['success'] += success
                index_results[index_type]['failed'] += failed
                
                print(f"🏁 워커 {worker_name} 최종 결과: 성공 {success}, 실패 {failed}")
            except queue.Empty:
                break
        
        # PostgreSQL 연결 종료
        if self.pg_connection:
            self.pg_connection.close()
        
        # 최종 통계
        total_time = time.time() - self.stats['start_time']
        success_rate = self.stats['successful'] / self.stats['total_processed'] * 100 if self.stats['total_processed'] > 0 else 0
        docs_per_hour = self.stats['successful'] / (total_time / 3600) if total_time > 0 else 0
        
        print(f"\n🎉 고속 3인덱스 병렬 처리 완료!")
        print(f"📊 Road 인덱스: {self.stats['successful']:,}개 성공")
        print(f"✅ 전체 성공률: {success_rate:.1f}%")
        print(f"⏱️ 총 소요시간: {total_time/3600:.2f}시간")
        print(f"🚀 평균 속도: {docs_per_hour:.0f} docs/hour")
        
        # 인덱스별 상세 결과
        print(f"\n📂 인덱스별 처리 결과:")
        for index_type, results in index_results.items():
            total_docs = results['success'] + results['failed']
            success_rate_idx = results['success'] / total_docs * 100 if total_docs > 0 else 0
            print(f"  {index_type.upper():>8}: {results['success']:,}개 성공 ({success_rate_idx:.1f}%)")
        
        return success_rate >= 70

    def analyze_query_type(self, query: str) -> str:
        """쿼리 타입 분석"""
        query_lower = query.lower()
        
        # 도로명 패턴
        road_patterns = ['로', '길', '대로', r'\d+번지', r'\d+-\d+']
        if any(pattern in query_lower or re.search(pattern, query_lower) for pattern in road_patterns):
            return 'road_focused'
        
        # 지번/동명 패턴
        jibun_patterns = ['동', '읍', '면', '리', '번지']
        if any(pattern in query_lower for pattern in jibun_patterns):
            return 'jibun_focused'
        
        # 빌딩명 패턴
        building_patterns = ['빌딩', '타워', '센터', '병원', '학교', '아파트', '호텔']
        if any(pattern in query_lower for pattern in building_patterns):
            return 'building_focused'
        
        return 'mixed'

    def smart_three_index_search(self, query: str, k: int = 5) -> List[Dict]:
        """3인덱스 스마트 검색"""
        
        print(f"\n🔍 3인덱스 스마트 검색: '{query}'")
        
        # 쿼리 타입 분석
        query_type = self.analyze_query_type(query)
        print(f"🧠 쿼리 타입: {query_type}")
        
        # 가중치 설정
        if query_type == 'road_focused':
            weights = {'road': 0.7, 'jibun': 0.2, 'building': 0.1}
        elif query_type == 'jibun_focused':
            weights = {'road': 0.2, 'jibun': 0.7, 'building': 0.1}
        elif query_type == 'building_focused':
            weights = {'road': 0.1, 'jibun': 0.1, 'building': 0.8}
        else:  # mixed
            weights = {'road': 0.4, 'jibun': 0.3, 'building': 0.3}
        
        print(f"⚖️ 가중치: {weights}")
        
        all_results = []
        
        # 각 인덱스에서 검색
        for index_type, weight in weights.items():
            if weight <= 0:
                continue
                
            try:
                results = self.vector_stores[index_type].similarity_search_with_score(
                    query, k=k*2  # 더 많이 가져와서 다양성 확보
                )
                
                print(f"  📋 {index_type}: {len(results)}개 결과")
                
                for doc, score in results:
                    # 가중치 적용한 점수 계산
                    weighted_score = score * weight
                    
                    result = {
                        'document': doc,
                        'original_score': score,
                        'weighted_score': weighted_score,
                        'source_index': index_type,
                        'weight': weight
                    }
                    all_results.append(result)
                    
            except Exception as e:
                print(f"  ❌ {index_type} 검색 오류: {e}")
        
        # 결과 통합 및 중복 제거
        unique_results = self.deduplicate_results(all_results)
        
        # 가중치 점수로 정렬
        unique_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # 상위 k개 반환
        final_results = unique_results[:k]
        
        print(f"  🎯 최종 결과: {len(final_results)}개 (중복 제거 후)")
        
        return final_results

    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """결과 중복 제거 (도로명주소 기준)"""
        seen_addresses = {}
        
        for result in results:
            road_addr = result['document'].metadata.get('road_address', '')
            
            if road_addr not in seen_addresses:
                seen_addresses[road_addr] = result
            else:
                # 기존 결과보다 점수가 높으면 교체
                if result['weighted_score'] > seen_addresses[road_addr]['weighted_score']:
                    seen_addresses[road_addr] = result
        
        return list(seen_addresses.values())

    def search_test(self, test_queries: List[str], k: int = 5):
        """3인덱스 검색 테스트"""
        print(f"\n🔍 3인덱스 검색 테스트 (Road + Jibun + Building)")
        print("=" * 70)
        
        if not self.vector_stores:
            self.initialize_vector_stores()
        
        for query in test_queries:
            print(f"\n🔎 검색어: '{query}'")
            print("-" * 50)
            
            try:
                results = self.smart_three_index_search(query, k=k)
                
                if not results:
                    print("  ❌ 검색 결과 없음")
                    continue
                
                print(f"📋 검색 결과: {len(results)}개")
                
                for i, result in enumerate(results):
                    doc = result['document']
                    road_addr = doc.metadata.get('road_address', '')
                    jibun_addr = doc.metadata.get('jibun_address', '')
                    building_name = doc.metadata.get('building_name', '')
                    source_index = result['source_index']
                    weighted_score = result['weighted_score']
                    original_score = result['original_score']
                    
                    print(f"\n  📍 {i+1}번째 결과 [출처: {source_index.upper()}]")
                    print(f"     🏠 도로명: {road_addr}")
                    if jibun_addr and jibun_addr != road_addr:
                        print(f"     📮 지번: {jibun_addr}")
                    if building_name:
                        print(f"     🏢 건물명: {building_name}")
                    
                    print(f"     🎯 가중치 점수: {weighted_score:.4f} (원본: {original_score:.4f})")
                    print(f"     📄 검색된 내용: {doc.page_content[:80]}...")
                        
            except Exception as e:
                print(f"  ❌ 검색 오류: {e}")

    def get_index_statistics(self):
        """인덱스별 통계 조회"""
        print(f"\n📊 3인덱스 통계")
        print("="*50)
        
        for index_type, index_name in self.indexes.items():
            try:
                response = requests.get(
                    f"{self.opensearch_url}/{index_name}/_count",
                    auth=self.auth,
                    verify=self.verify_ssl
                )
                
                if response.status_code == 200:
                    count = response.json().get('count', 0)
                    print(f"📂 {index_type.upper():>8}: {count:,}개 문서")
                else:
                    print(f"❌ {index_type.upper():>8}: 조회 실패")
                    
            except Exception as e:
                print(f"❌ {index_type.upper():>8}: 오류 - {e}")

    def benchmark_search_methods(self, test_queries: List[str], k: int = 3):
        """검색 방법별 성능 벤치마크"""
        print(f"\n🏃‍♂️ 3인덱스 검색 방법별 성능 벤치마크")
        print("=" * 70)
        
        methods = [
            ('Road 단독', lambda q: self.vector_stores['road'].similarity_search(q, k=k)),
            ('Jibun 단독', lambda q: self.vector_stores['jibun'].similarity_search(q, k=k)),
            ('Building 단독', lambda q: self.vector_stores['building'].similarity_search(q, k=k)),
            ('3인덱스 스마트 검색', lambda q: [r['document'] for r in self.smart_three_index_search(q, k=k)])
        ]
        
        for query in test_queries:
            print(f"\n🔎 테스트 쿼리: '{query}'")
            print("-" * 40)
            
            for method_name, search_func in methods:
                try:
                    start_time = time.time()
                    results = search_func(query)
                    search_time = time.time() - start_time
                    
                    print(f"  📋 {method_name:15}")
                    print(f"    ⏱️ 검색 시간: {search_time*1000:.1f}ms")
                    print(f"    📊 결과 수: {len(results)}개")
                    
                    if results:
                        if hasattr(results[0], 'metadata'):
                            first_result = results[0].metadata.get('road_address', '')[:50]
                        else:
                            first_result = str(results[0])[:50]
                        print(f"    🥇 1순위: {first_result}...")
                    
                except Exception as e:
                    print(f"  ❌ {method_name}: 오류 - {str(e)[:30]}")


def main():
    """메인 실행 함수"""
    print("🇰🇷 최종 3인덱스 고속 병렬 RAG 시스템")
    print("=" * 70)
    print("🔧 구조:")
    print("  • Road 인덱스: 도로명주소 전용 (빌딩명 제외)")
    print("  • Jibun 인덱스: 지번주소 전용")
    print("  • Building 인덱스: 빌딩명+지역 (aliases 제거)")
    print("=" * 70)
    
    # 시스템 초기화
    rag_system = FinalThreeIndexRAG()
    
    # OpenSearch 연결 확인
    if not rag_system.check_opensearch_connection():
        print("❌ OpenSearch 연결 실패")
        return
    
    print("\n📋 실행 옵션:")
    print("1. 🔥 고속 병렬 처리 (3인덱스 생성 + 병렬 처리)")
    print("2. 🔍 검색 테스트 (기존 인덱스 사용)")
    print("3. 📊 인덱스 통계 확인")
    print("4. 🏃‍♂️ 검색 성능 벤치마크")
    print("5. 🧹 모든 인덱스 삭제")
    print("6. 종료")
    
    choice = input("\n선택 (1-6): ").strip()
    
    if choice == "1":
        # 고속 병렬 처리
        if not rag_system.create_three_indexes():
            print("❌ 인덱스 생성 실패")
            return
        
        print("\n⚙️ 고속 병렬 처리 설정:")
        workers = int(input("워커 수 (기본 6, 권장 4-8): ") or "6")
        data_batch = int(input("데이터 배치 (기본 3000): ") or "3000")
        index_batch = int(input("인덱싱 배치 (기본 100): ") or "100")
        
        print(f"\n🚀 고속 병렬 처리 시작!")
        print(f"📊 설정: 워커 {workers}개, 데이터배치 {data_batch}, 인덱싱배치 {index_batch}")
        print(f"⚡ 예상 처리속도: 15,000-30,000 docs/hour")
        
        success = rag_system.high_speed_parallel_processing(
            data_batch_size=data_batch,
            indexing_batch_size=index_batch,
            num_workers=workers
        )
        
        if success:
            print("\n✅ 고속 처리 성공!")
            rag_system.get_index_statistics()
            
            # 자동 검색 테스트
            print("\n🔍 자동 검색 테스트 시작...")
            test_queries = [
                "테헤란로2길 21",          # Road 중심
                "역삼동 825-27",           # Jibun 중심
                "세브란스병원",            # Building 중심
                "서울 삼성빌딩",           # Building + 지역
                "강남구 테헤란로"          # Road + 지역
            ]
            
            rag_system.search_test(test_queries, k=3)
            
        else:
            print("❌ 고속 처리 실패")
    
    elif choice == "2":
        print("\n🔍 검색 테스트 모드")
        
        test_queries = []
        print("\n검색어를 입력하세요 (엔터만 치면 기본 검색어 사용):")
        
        while True:
            query = input("검색어: ").strip()
            if not query:
                break
            test_queries.append(query)
        
        if not test_queries:
            test_queries = [
                "테헤란로2길 21",          # 도로명주소
                "역삼동 825",              # 지번주소
                "세브란스병원",            # 빌딩명
                "서울 삼성빌딩",           # 지역+빌딩
                "강남구 테헤란로",         # 지역+도로명
                "부산 롯데호텔",           # 지역+빌딩
                "연세로 50-1",             # 도로명+번지
                "신촌동"                   # 동명
            ]
            print("기본 검색어 사용")
        
        rag_system.search_test(test_queries, k=5)
    
    elif choice == "3":
        rag_system.get_index_statistics()
    
    elif choice == "4":
        print("\n🏃‍♂️ 검색 성능 벤치마크")
        
        benchmark_queries = [
            "테헤란로2길 21",          # 도로명 정확 검색
            "세브란스병원",            # 빌딩명 검색
            "역삼동",                 # 동명 검색
            "서울 삼성",              # 지역+빌딩 부분검색
            "강남구 병원",            # 지역+카테고리
        ]
        
        rag_system.benchmark_search_methods(benchmark_queries)

    elif choice == "5":
        confirm = input("⚠️ 모든 인덱스를 삭제하시겠습니까? (y/N): ").strip().lower()
        if confirm == 'y':
            for index_type, index_name in rag_system.indexes.items():
                rag_system._delete_index_if_exists(index_name)
            print("✅ 모든 인덱스 삭제 완료")
        else:
            print("❌ 삭제 취소")
    
    elif choice == "6":
        print("👋 종료합니다.")
        return
    
    else:
        print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()