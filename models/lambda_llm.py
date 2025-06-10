from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Any, Dict, List, Mapping, Optional, Iterator
from logger import logger
from openai import OpenAI
import asyncio

class LambdaAILLM(LLM):
    """Lambda.ai API를 사용하기 위한 커스텀 LLM 구현 (OpenAI 클라이언트 사용)"""
    
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1024
    api_base: str = "https://api.lambda.ai/v1"
    
    @property
    def _llm_type(self) -> str:
        return "lambda_ai"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Lambda.ai API를 호출하여 텍스트 생성"""
        try:
            # OpenAI 클라이언트 초기화
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            
            logger.debug(f"Lambda.ai API 호출 준비: 모델={self.model}")
            
            # 추가 옵션 설정
            options = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            # 중지 토큰 추가
            if stop:
                options["stop"] = stop
                
            # 추가 매개변수 적용
            for k, v in kwargs.items():
                if k not in ["system_prompt"]:  # system_prompt는 메시지에서 처리
                    options[k] = v
            
            # 시스템 프롬프트 확인
            system_prompt = kwargs.pop("system_prompt", None)
            messages = []
            
            # 시스템 프롬프트가 있으면 추가
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # 사용자 프롬프트 추가
            messages.append({"role": "user", "content": prompt})
            
            # API 호출
            response = client.chat.completions.create(
                messages=messages,
                **options
            )
            
            # 응답에서 텍스트 추출
            return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Lambda.ai API 호출 중 오류 발생: {str(e)}")
            raise ValueError(f"Lambda.ai API 호출 중 오류 발생: {str(e)}")
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """모델을 식별하기 위한 파라미터 반환"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.api_base
        }


class LambdaAIChatModel(BaseChatModel):
    """Lambda.ai API를 사용하기 위한 Chat 모델 구현 (메시지 히스토리 지원)"""
    
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1024
    api_base: str = "https://api.lambda.ai/v1"
    
    @property
    def _llm_type(self) -> str:
        return "lambda_ai_chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """메시지 리스트를 받아서 Chat 응답 생성"""
        try:
            # OpenAI 클라이언트 초기화
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            
            logger.debug(f"Lambda.ai Chat API 호출 준비: 모델={self.model}, 메시지 수={len(messages)}")
            
            # LangChain 메시지를 OpenAI 형식으로 변환
            openai_messages = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    openai_messages.append({"role": "system", "content": message.content})
                elif isinstance(message, HumanMessage):
                    openai_messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    openai_messages.append({"role": "assistant", "content": message.content})
                else:
                    # BaseMessage의 기본 처리
                    role = getattr(message, "role", "user")
                    openai_messages.append({"role": role, "content": message.content})
            
            # 추가 옵션 설정
            options = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            # 중지 토큰 추가
            if stop:
                options["stop"] = stop
                
            # 추가 매개변수 적용
            for k, v in kwargs.items():
                if k not in ["messages"]:  # messages는 이미 처리됨
                    options[k] = v
            
            # API 호출
            response = client.chat.completions.create(
                messages=openai_messages,
                **options
            )
            
            # 응답에서 텍스트 추출
            content = response.choices[0].message.content
            
            # ChatGeneration 객체 생성
            generation = ChatGeneration(
                message=AIMessage(content=content),
                generation_info={
                    "model": self.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": response.usage.dict() if response.usage else None
                }
            )
            
            return ChatResult(generations=[generation])
                
        except Exception as e:
            logger.error(f"Lambda.ai Chat API 호출 중 오류 발생: {str(e)}")
            raise ValueError(f"Lambda.ai Chat API 호출 중 오류 발생: {str(e)}")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """비동기 메시지 생성"""
        # 동기 메서드를 비동기로 래핑
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self._generate(messages, stop, run_manager, **kwargs)
        )
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """모델을 식별하기 위한 파라미터 반환"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.api_base
        }


class LambdaImageAnalyzer:
    """Lambda.ai의 비전 모델을 사용하여 이미지를 분석하는 클래스"""
    
    def __init__(self, api_key, model, api_base="https://api.lambda.ai/v1"):
        """초기화 함수"""
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    def analyze_image(
        self,
        prompt: str,
        image_url: str,
        system_prompt: str = "당신은 이미지를 분석하는 전문가입니다. 이미지의 내용을 정확하고 상세하게 설명해주세요.",
        temperature: float = 0.5,
        max_tokens: int = 1000,
        return_raw_response: bool = False,
        **kwargs
    ):
        """이미지를 분석하고 텍스트 설명을 생성"""
        try:
            logger.debug(f"Lambda.ai 비전 API 호출 준비: 모델={self.model}")
            
            # 멀티모달 메시지 구성
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ]
            
            # 추가 옵션 설정
            options = {
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            # API 호출
            response = self.client.chat.completions.create(
                messages=messages,
                **options
            )
            
            # 응답에서 텍스트 추출
            analysis = response.choices[0].message.content
            
            result = {
                "analysis": analysis,
                "model_info": self.model,
            }
            
            # 원시 응답을 요청한 경우 포함
            if return_raw_response:
                result["raw_response"] = response
                
            return result
                
        except Exception as e:
            logger.error(f"Lambda.ai 비전 API 호출 중 오류 발생: {str(e)}")
            raise ValueError(f"Lambda.ai 비전 API 호출 중 오류 발생: {str(e)}")


# LangChain과 통합하기 위한 래퍼 함수들
def get_lambda_llm(api_key, model, **kwargs):
    """LangChain과 호환되는 LambdaAILLM 인스턴스 생성 (기존 호환용)"""
    return LambdaAILLM(
        api_key=api_key,
        model=model,
        **kwargs
    )

def get_lambda_chat_model(api_key, model, **kwargs):
    """LangChain과 호환되는 LambdaAIChatModel 인스턴스 생성 (메시지 히스토리 지원)"""
    return LambdaAIChatModel(
        api_key=api_key,
        model=model,
        **kwargs
    )

def analyze_image_with_lambda(prompt, image_url, api_key, model, **kwargs):
    """Lambda.ai의 비전 모델을 사용하여 이미지 분석"""
    analyzer = LambdaImageAnalyzer(
        api_key=api_key,
        model=model,
        api_base=kwargs.pop("api_base", "https://api.lambda.ai/v1")
    )
    return analyzer.analyze_image(prompt=prompt, image_url=image_url, **kwargs)

# 사용 예시
async def example_lambda_usage():
    """Lambda AI 모델 사용 예시"""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Chat 모델 초기화
    chat_model = get_lambda_chat_model(
        api_key="your_lambda_api_key",
        model="your_lambda_model",
        temperature=0.7
    )
    
    # 메시지 리스트 생성
    messages = [
        SystemMessage(content="당신은 도움이 되는 AI 어시스턴트입니다."),
        HumanMessage(content="안녕하세요! Lambda AI에 대해 알려주세요.")
    ]
    
    # 응답 생성
    result = chat_model._generate(messages)
    print(f"Lambda AI 응답: {result.generations[0].message.content}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_lambda_usage())