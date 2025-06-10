from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import os
import base64
from config import OPENAI_MODEL, OPENAI_API_KEY, GEMINI_MODEL, GEMINI_API_KEY, LAMBDA_API_KEY, LAMBDA_MODEL, LAMBDA_IMG_MODEL, NAVER_CLOVA_API_KEY
from logger import logger

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
from models.lambda_llm import LambdaAILLM, get_lambda_chat_model, analyze_image_with_lambda

# 새로운 메시지 히스토리 시스템 임포트
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from typing import List, Dict, Optional
from pydantic import Field, BaseModel
from dataclasses import dataclass
from threading import Lock
import asyncio

# ================== 새로운 LangChain 메시지 히스토리 시스템 ==================

class SessionedInMemoryChatMessageHistory:
    """세션별 인메모리 채팅 메시지 히스토리 관리자"""
    
    def __init__(self, session_timeout_hours: int = 1):
        self.store: Dict[str, InMemoryChatMessageHistory] = {}
        self.last_accessed: Dict[str, float] = {}
        self.session_timeout_seconds = session_timeout_hours * 3600    # 3600초 = 1시간
        self.lock = Lock()
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """세션 ID에 해당하는 메시지 히스토리 반환"""
        with self.lock:
            current_time = time.time()
            
            # 세션이 존재하지 않으면 새로 생성
            if session_id not in self.store:
                self.store[session_id] = InMemoryChatMessageHistory()
                self.last_accessed[session_id] = current_time
                logger.debug(f"새 세션 생성: {session_id}")
            else:
                # 세션 만료 확인
                if current_time - self.last_accessed[session_id] > self.session_timeout_seconds:
                    logger.info(f"세션 만료로 인한 초기화: {session_id}")
                    self.store[session_id] = InMemoryChatMessageHistory()
                
                # 마지막 접근 시간 업데이트
                self.last_accessed[session_id] = current_time
            
            return self.store[session_id]
    
    def clear_session(self, session_id: str) -> bool:
        """특정 세션 삭제"""
        with self.lock:
            if session_id in self.store:
                del self.store[session_id]
                if session_id in self.last_accessed:
                    del self.last_accessed[session_id]
                logger.info(f"세션 삭제됨: {session_id}")
                return True
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """만료된 세션들 정리"""
        current_time = time.time()
        expired_sessions = []
        
        with self.lock:
            for session_id, last_access in self.last_accessed.items():
                if current_time - last_access > self.session_timeout_seconds:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                if session_id in self.store:
                    del self.store[session_id]
                del self.last_accessed[session_id]
        
        if expired_sessions:
            logger.info(f"만료된 세션 {len(expired_sessions)}개 정리 완료")
        
        return len(expired_sessions)
    
    def get_conversation_summary(self, session_id: str, max_messages: int = 10) -> str:
        """대화 히스토리를 문자열로 요약"""
        history = self.get_session_history(session_id)
        messages = history.messages[-max_messages:] if max_messages > 0 else history.messages
        
        formatted_history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_history.append(f"사용자: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_history.append(f"상담원: {msg.content}")
        
        return "\n".join(formatted_history)

# 전역 메시지 히스토리 관리자
message_history_manager = SessionedInMemoryChatMessageHistory(session_timeout_hours=1)

# BaseRetriever를 상속하고 Pydantic 필드 사용
class CustomRetriever(BaseRetriever):
    documents: List[Document] = Field(default_factory=list)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 간단히 모든 문서 반환
        return self.documents

# 검색된 문서를 컨텍스트로 결합
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def create_session_id(user_id: str, llm_type: str) -> str:
    """사용자 ID와 LLM 타입으로 세션 ID 생성"""
    return f"{user_id}_{llm_type.lower()}"

async def generate_rag_response(request_id, query, docs, llm_name, user_id=None):
    """
    검색된 문서를 기반으로 생성형 AI 응답을 생성합니다. (최신 LangChain 메시지 히스토리 사용)
    
    Args:
        request_id (str): 요청 식별자
        query (str): 사용자 질문
        docs (List[Document]): 검색된 문서 리스트
        llm_name (str): 사용할 LLM 이름
        user_id (str, optional): 사용자 ID (히스토리 기능용)
        
    Returns:
        dict: 생성된 응답과 소스 문서를 포함하는 딕셔너리
    """
    try:
        logger.info(f"[{request_id}] {llm_name} LLM으로 응답 생성 시작")

        # 문서들을 텍스트로 변환
        context = format_docs(docs)
        logger.debug(f"[{request_id}] context : {context}")

        gen_start_time = time.time()

        # LLM 초기화
        if llm_name.lower() == "gemini":
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key= GEMINI_API_KEY,
                temperature=0.7
            )
        elif llm_name.lower() == "chatgpt":
            llm = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0.7
            )
        elif llm_name.lower() == "llama":
            llm = get_lambda_chat_model(
                model=LAMBDA_MODEL,
                api_key=LAMBDA_API_KEY,
                temperature=0.7,
                max_tokens=1024
            )            
        else:
            raise ValueError(f"지원하지 않는 LLM: {llm_name}")        

        # 대화 히스토리가 있는 경우와 없는 경우 구분
        if user_id:
            session_id = create_session_id(user_id, llm_name)
            
            # 대화 히스토리 조회
            conversation_history = message_history_manager.get_conversation_summary(session_id, max_messages=10)
            logger.info(f"[{request_id}] conversation_history : {conversation_history}")
            
            # 히스토리가 있는 경우의 프롬프트
            if conversation_history:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """당신은 콜마너 대리운전 상담원입니다.
제공된 정보와 이전 대화 내용을 바탕으로 질문에 정확하고 친절하게 답변해 주세요.
이전 대화 내용도 참고해서 안내하고 사용자의 질문과 컨텍스트 내용이 관련이 없으면 "관련 내용 없음" 이라고 안내해줘

관련 정보:
{context}

이전 대화 내용:
{conversation_history}"""),
                    ("human", "{input}")
                ])
            else:
                # 히스토리가 없는 경우의 프롬프트
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """당신은 콜마너 대리운전 상담원입니다.
제공된 정보를 바탕으로 질문에 정확하고 친절하게 답변해 주세요.
사용자의 질문과 컨텍스트 내용이 관련이 없으면 "관련 내용 없음" 이라고 안내해줘

관련 정보:
{context}"""),
                    ("human", "{input}")
                ])
            
            # 메시지 히스토리를 포함한 체인 생성
            chain = prompt | llm | StrOutputParser()
            
            # RunnableWithMessageHistory로 래핑
            chain_with_history = RunnableWithMessageHistory(
                chain,
                message_history_manager.get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )
            
            # 체인 실행
            if conversation_history:
                result = chain_with_history.invoke(
                    {
                        "input": query,
                        "context": context,
                        "conversation_history": conversation_history
                    },
                    config={"configurable": {"session_id": session_id}}
                )
            else:
                result = chain_with_history.invoke(
                    {
                        "input": query,
                        "context": context
                    },
                    config={"configurable": {"session_id": session_id}}
                )
        else:
            # 히스토리 없는 단순 체인
            prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 콜마너 대리운전 상담원입니다.
제공된 정보를 바탕으로 질문에 정확하고 친절하게 답변해 주세요.
사용자의 질문과 컨텍스트 내용이 관련이 없으면 "관련 내용 없음" 이라고 안내해줘

관련 정보:
{context}"""),
                ("human", "{input}")
            ])
            
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"input": query, "context": context})
        
        gen_end_time = time.time()
        gen_duration = round((gen_end_time - gen_start_time) * 1000, 2)
        logger.info(f"[{request_id}] 응답 생성 완료: 소요시간={gen_duration}ms")
        
        # 결과 처리
        answer = result if isinstance(result, str) else str(result)
        logger.debug(f"[{request_id}] 생성된 응답: {answer}")
        
        response_data = {
            "answer": answer,
            "llm_type": llm_name.lower(),
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs
            ]
        }
        
        # 대화 히스토리 정보 추가 (user_id가 있는 경우)
        if user_id:
            session_id = create_session_id(user_id, llm_name)
            history = message_history_manager.get_session_history(session_id)
            response_data["conversation_history"] = [
                {
                    "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content,
                    "llm_type": llm_name.lower()
                } for msg in history.messages
            ]
        
        return response_data
        
    except Exception as e:
        logger.error(f"[{request_id}] 생성형 응답 중 오류 발생: {e}")
        # 오류가 발생한 경우 검색 결과만 반환
        return {
            "answer": f"생성형 응답 중 오류가 발생했습니다: {str(e)}",
            "llm_type": llm_name.lower() if llm_name else None,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs
            ]
        }

async def analyze_image_url_with_model(request_id, image_url, img_base64, description, model_type):
    """URL을 통해 특정 모델을 사용하여 이미지를 분석합니다."""  
    
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
            client = OpenAI(api_key=OPENAI_API_KEY,
                            timeout=30.0  # 30초로 설정)
            )
            
            response = client.chat.completions.create(
                model=OPENAI_MODEL,  # 또는 최신 비전 모델
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
            model_info = OPENAI_MODEL
            
        elif model_type.lower() == "gemini":
            # Google의 Gemini Pro Vision 사용 (URL 직접 전달)
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            
            genai.configure(api_key=GEMINI_API_KEY)
            
            # 안전 설정
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
            
            # Gemini Pro Vision 모델
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                safety_settings=safety_settings
            )
            
            # base64 인코딩된 이미지 사용
            response = model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": img_base64}
            ])
            
            analysis = response.text
            model_info = GEMINI_MODEL
        elif model_type.lower() == "llama":

            response = analyze_image_with_lambda(
                api_key=LAMBDA_API_KEY,
                model=LAMBDA_IMG_MODEL,
                prompt=prompt,
                image_url=image_url
            )
            
            analysis = response['analysis']
            model_info = LAMBDA_IMG_MODEL
            
        elif model_type.lower() == "naver":

            from models.naver_llm import ClovaStudioAPI

            CLOVA_API_KEY = NAVER_CLOVA_API_KEY  # 클로바 스튜디오 API 키
            
            # 클로버 스튜디오 API 인스턴스 생성
            clova = ClovaStudioAPI(CLOVA_API_KEY)  

            # === cURL 예제와 동일한 방식으로 시스템 메시지 생성 ===
            system_message = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "당신은 꽃집에서 일하는 상담원이고 해당 이미지는 손글씨로 이미지에서 전화번호를 포함한 모든 텍스트를 정확하게 추출해주세요."
                    }
                ]
            }
            
            user_message = clova.create_vision_message(
        #        text="이 이미지에 포함된 모든 텍스트에서 보내신분, 품명, 수량, 경.조사어, 배달일시, 배달장소, 기타, 받으신분, 인수하신분, 서명까지 모두 정확히 추출해주세요. 전화번호도 있으면 마스킹으로 처리해주세요. 텍스트가 없다면 '텍스트 없음'이라고 답변해주세요.",
                text="이 이미지에 포함된 모든 정보를 추출합니다. 추출된 정보를 목록 형태로 정리합니다.",
                image_url=image_url
            )
            
            messages = [system_message, user_message]

            response = clova.chat_completions_v3(
                messages=messages,
                model_name="HCX-005",
                top_p=0.8,
                top_k=0,
                max_tokens=100,
                temperature=0.5,
                repetition_penalty=1.1,
                stop=[]
            )                      

            analysis = response['analysis']
            model_info = "HCX-005"

        else:
            raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        logger.info(f"[{request_id}] {model_type} URL 이미지 분석 완료: 소요시간={processing_time}ms, analysis={analysis}")
        
        return {
            "analysis": analysis,
            "model": model_info,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.exception(f"[{request_id}] {model_type} URL 이미지 분석 중 오류: {e}")
        raise Exception(f"{model_type} URL 이미지 분석 중 오류: {str(e)}")

# ================== 히스토리 관리 유틸리티 함수 ==================

def clear_user_session(user_id: str, llm_type: str = None) -> bool:
    """특정 사용자의 세션을 삭제합니다."""
    if llm_type:
        session_id = create_session_id(user_id, llm_type)
        return message_history_manager.clear_session(session_id)
    else:
        # 모든 LLM 타입의 세션 삭제
        llm_types = ["gemini", "chatgpt", "llama"]
        success_count = 0
        for llm in llm_types:
            session_id = create_session_id(user_id, llm)
            if message_history_manager.clear_session(session_id):
                success_count += 1
        return success_count > 0

def get_user_conversation_history(user_id: str, llm_type: str, max_messages: int = 10) -> str:
    """특정 사용자의 대화 히스토리를 문자열로 반환합니다."""
    session_id = create_session_id(user_id, llm_type)
    return message_history_manager.get_conversation_summary(session_id, max_messages)

def get_user_messages(user_id: str, llm_type: str) -> List[Dict]:
    """특정 사용자의 메시지 객체들을 딕셔너리 리스트로 반환합니다."""
    session_id = create_session_id(user_id, llm_type)
    history = message_history_manager.get_session_history(session_id)
    
    return [
        {
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content,
            "llm_type": llm_type.lower()
        } for msg in history.messages
    ]

def cleanup_all_expired_sessions() -> int:
    """모든 만료된 세션을 정리합니다."""
    return message_history_manager.cleanup_expired_sessions()

# ================== 사용 예시 ==================

async def example_usage():
    """최신 LangChain 메시지 히스토리 사용 예시"""
    request_id = "test_001"
    user_id = "user123"
    
    # 가상의 문서 생성
    docs = [
        type('Document', (), {'page_content': '대리운전 기본요금은 13,000원입니다.', 'metadata': {'source': 'price_guide'}})(),
        type('Document', (), {'page_content': '거리 추가 요금은 km당 1,000원입니다.', 'metadata': {'source': 'price_guide'}})()
    ]
    
    print("=== 최신 LangChain 메시지 히스토리를 사용한 RAG 테스트 ===")
    
    # 첫 번째 질문 (Gemini 사용)
    response1 = await generate_rag_response(
        request_id=request_id,
        query="대리운전 요금이 얼마인가요?",
        docs=docs,
        llm_name="gemini",
        user_id=user_id
    )
    print(f"첫 번째 응답 (Gemini): {response1['answer']}")
    
    # 두 번째 질문 (같은 LLM 사용)
    response2 = await generate_rag_response(
        request_id=request_id,
        query="10km 가면 총 얼마나 나와요?",
        docs=docs,
        llm_name="gemini",
        user_id=user_id
    )
    print(f"두 번째 응답 (Gemini): {response2['answer']}")
    
    # 대화 히스토리 조회
    history = get_user_conversation_history(user_id, "gemini", max_messages=10)
    print(f"\n대화 히스토리:\n{history}")
    
    # 메시지 객체 조회
    messages = get_user_messages(user_id, "gemini")
    print(f"\n메시지 객체: {len(messages)}개")
    for i, msg in enumerate(messages):
        print(f"{i+1}. [{msg['role']}] {msg['content']}")
    
    # 세션 정리 테스트
    print(f"\n세션 삭제 결과: {clear_user_session(user_id, 'gemini')}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())