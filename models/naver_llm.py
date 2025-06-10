import requests
import json
import uuid
import base64
from PIL import Image
import io
from urllib.parse import urlparse
from config import NAVER_CLOVA_API_KEY

class ClovaStudioAPI:
    def __init__(self, clova_api_key):
        """
        클로버 스튜디오 API 클래스 초기화
        
        Args:
            clova_api_key (str): 클로바 스튜디오 API 키
            apigw_api_key (str, optional): API Gateway API 키 (필요한 경우)
        """
        self.clova_api_key = clova_api_key
        self.base_url = "https://clovastudio.stream.ntruss.com"
    
    def _get_headers(self):
        """API 호출용 헤더 생성"""
        headers = {
            "Authorization": f"Bearer {self.clova_api_key}",
#            "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
            "Content-Type": "application/json"
#            "Accept": "text/event-stream"
        }
        
        return headers
    
    def chat_completions_v3(self, messages, model_name="HCX-005", **kwargs):
        """
        Chat Completions v3 API 호출 (이미지 지원)
        
        Args:
            messages (list): 대화 메시지 리스트 (텍스트 + 이미지 지원)
            model_name (str): 모델 이름 (HCX-005: 비전모델, HCX-DASH-002: 텍스트전용)
            **kwargs: 추가 파라미터들
        
        Returns:
            dict: API 응답 결과
        """
        url = f"{self.base_url}/testapp/v3/chat-completions/{model_name}"
        headers = self._get_headers()
        
        # 기본 파라미터 설정 (cURL 예제와 동일한 구조)
        data = {
            "messages": messages,
            "topP": kwargs.get("top_p", 0.8),
            "topK": kwargs.get("top_k", 0),
            "maxTokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.5),
            "repetitionPenalty": kwargs.get("repetition_penalty", 1.1),
            "stop": kwargs.get("stop", [])
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            analysis = response.json()

            result = {
                "analysis": analysis["result"]["message"]["content"],
                "model_info": model_name,
            }

            return result
        except requests.exceptions.RequestException as e:
            return {"error": f"API 호출 실패: {str(e)}"}
    
    def encode_image_to_base64(self, image_path, max_size=(1024, 1024)):
        """
        이미지를 base64로 인코딩하고 크기 조정
        
        Args:
            image_path (str): 이미지 파일 경로
            max_size (tuple): 최대 크기 (width, height)
        
        Returns:
            str: base64 인코딩된 이미지 문자열
        """
        try:
            # 이미지 열기
            with Image.open(image_path) as img:
                # RGBA를 RGB로 변환 (필요한 경우)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # 이미지 크기 조정 (비율 유지)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 메모리 버퍼에 이미지 저장
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)
                
                # base64 인코딩
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return f"data:image/jpeg;base64,{image_base64}"
                
        except Exception as e:
            raise Exception(f"이미지 인코딩 실패: {str(e)}")
    
    def download_image_from_url(self, image_url, max_size=(1024, 1024)):
        """
        웹 URL에서 이미지를 다운로드하고 base64로 인코딩
        
        Args:
            image_url (str): 이미지 웹 URL (http:// 또는 https://)
            max_size (tuple): 최대 크기 (width, height)
        
        Returns:
            str: base64 인코딩된 이미지 문자열
        """
        try:
            # URL 유효성 검사
            parsed_url = urlparse(image_url)
            if not parsed_url.scheme or parsed_url.scheme not in ['http', 'https']:
                raise ValueError("올바른 HTTP/HTTPS URL을 입력해주세요.")
            
            # 이미지 다운로드
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Content-Type 확인
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"올바른 이미지가 아닙니다. Content-Type: {content_type}")
            
            # 이미지 처리
            image_data = response.content
            with Image.open(io.BytesIO(image_data)) as img:
                # RGBA를 RGB로 변환 (필요한 경우)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # 이미지 크기 조정 (비율 유지)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 메모리 버퍼에 이미지 저장
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)
                
                # base64 인코딩
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return f"data:image/jpeg;base64,{image_base64}"
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"이미지 다운로드 실패: {str(e)}")
        except Exception as e:
            raise Exception(f"이미지 처리 실패: {str(e)}")

    def create_vision_message(self, text, image_path=None, image_base64=None, image_url=None, use_data_uri=False):
        """
        비전 메시지 생성 (텍스트 + 이미지) - 네이버 클로바 스튜디오 API 문서 기준
        
        Args:
            text (str): 텍스트 내용
            image_path (str, optional): 이미지 파일 경로
            image_base64 (str, optional): base64 인코딩된 이미지
            image_url (str, optional): 이미지 웹 URL
            use_data_uri (bool): True시 dataUri 방식, False시 imageUrl 방식 사용
        
        Returns:
            dict: 메시지 객체
        """
        image_inputs = [image_path, image_base64, image_url]
        provided_inputs = [inp for inp in image_inputs if inp is not None]
        
        if len(provided_inputs) > 1:
            raise ValueError("image_path, image_base64, image_url 중 하나만 제공해야 합니다.")
        
        content = []
        
        # 이미지 먼저 추가 (문서 예제 순서와 동일)
        if image_path:
            image_data = self.encode_image_to_base64(image_path)
            if use_data_uri:
                # dataUri 방식 (base64 데이터)
                base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                content.append({
                    "type": "dataUri",
                    "dataUri": {
                        "data": base64_data
                    }
                })
            else:
                # imageUrl 방식 (data URI 포함)
                content.append({
                    "type": "image_url",
                    "imageUrl": {
                        "url": image_data
                    }
                })
        elif image_base64:
            if use_data_uri:
                # dataUri 방식 (순수 base64 데이터만)
                base64_data = image_base64.split(',')[1] if ',' in image_base64 else image_base64
                content.append({
                    "type": "dataUri", 
                    "dataUri": {
                        "data": base64_data
                    }
                })
            else:
                # imageUrl 방식 (data URI 형태)
                data_uri = image_base64 if image_base64.startswith('data:') else f"data:image/jpeg;base64,{image_base64}"
                content.append({
                    "type": "image_url",
                    "imageUrl": {
                        "url": data_uri
                    }
                })
        elif image_url:
            # 웹 URL은 항상 imageUrl 방식으로 처리
            content.append({
                "type": "image_url",
                "imageUrl": {
                    "url": image_url
                }
            })
        
        # 텍스트 추가
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        return {
            "role": "user",
            "content": content
        }

    def create_vision_message_with_url(self, text, image_url):
        """
        웹 URL 이미지를 사용한 비전 메시지 생성 (편의 메서드)
        
        Args:
            text (str): 텍스트 내용
            image_url (str): 이미지 웹 URL
        
        Returns:
            dict: 메시지 객체
        """
        return self.create_vision_message(text=text, image_url=image_url)
    
    # ===== 새로 추가된 웹 이미지 분석 편의 메서드들 =====
    
    def analyze_web_image(self, image_url, question="이 이미지에 무엇이 보이나요? 자세히 설명해주세요.", system_prompt=None, verbose=True, **kwargs):
        """
        웹 URL 이미지를 분석하는 완전한 워크플로우
        
        Args:
            image_url (str): 분석할 이미지 URL
            question (str): 이미지에 대한 질문
            system_prompt (str, optional): 시스템 프롬프트 (기본값 사용시 None)
            verbose (bool): 진행 상황 출력 여부
            **kwargs: API 파라미터들
        
        Returns:
            dict: API 응답 결과
        """
        try:
            if verbose:
                print("🔍 웹 이미지 분석 시작...")
                print(f"📎 이미지 URL: {image_url}")
            
            # 기본 시스템 프롬프트
            if system_prompt is None:
                system_prompt = "당신은 이미지를 자세히 분석하고 설명하는 전문 AI 어시스턴트입니다. 한국어로 정확하고 자세하게 답변해주세요."
            
            # 비전 메시지 생성
            vision_message = self.create_vision_message(question, image_url=image_url)
            
            messages = [
                {"role": "system", "content": system_prompt},
                vision_message
            ]
            
            if verbose:
                print("🤖 AI 모델 분석 중...")
            
            # API 호출
            result = self.chat_completions_v3(
                messages=messages,
                model_name="HCX-005",
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000),
                top_p=kwargs.get("top_p", 0.8),
                top_k=kwargs.get("top_k", 0),
                repetition_penalty=kwargs.get("repetition_penalty", 1.1),
                stop=kwargs.get("stop", [])
            )
            
            if verbose:
                if "error" not in result:
                    print("✅ 분석 완료!")
                else:
                    print(f"❌ 분석 실패: {result.get('error')}")
            
            return result
            
        except Exception as e:
            error_result = {"error": f"이미지 분석 실패: {str(e)}"}
            if verbose:
                print(f"❌ 오류 발생: {str(e)}")
            return error_result
    

    
    def extract_text_from_web_image(self, image_url, **kwargs):
        """
        웹 이미지에서 텍스트 추출 (OCR)
        
        Args:
            image_url (str): 텍스트를 추출할 이미지 URL
            **kwargs: API 파라미터들
        
        Returns:
            dict: API 응답 결과
        """
        return self.analyze_web_image(
            image_url=image_url,
            question="이 이미지에 포함된 모든 텍스트에서 이름, 전화번호, 주소를 인식하여 모두 정확히 추출해주세요. 텍스트가 없다면 '텍스트 없음'이라고 답변해주세요.",
            system_prompt="당신은 이미지에서 텍스트를 정확하게 추출하는 OCR 전문가입니다. 모든 텍스트를 빠짐없이 추출해주세요.",
            verbose=False,
            **kwargs
        )
    
    def create_vision_message_with_downloaded_image(self, text, image_url, max_size=(1024, 1024), use_data_uri=True):
        """
        웹 URL 이미지를 다운로드하여 base64로 인코딩 후 비전 메시지 생성
        (URL 직접 전송이 아닌 다운로드 후 파일 데이터로 전송)
        
        Args:
            text (str): 텍스트 내용
            image_url (str): 다운로드할 이미지 웹 URL
            max_size (tuple): 최대 크기 (width, height)
            use_data_uri (bool): True시 dataUri 방식, False시 imageUrl 방식으로 base64 전송
        
        Returns:
            dict: 메시지 객체
        """
        try:
            # 웹 이미지를 다운로드하고 base64로 인코딩
            image_base64 = self.download_image_from_url(image_url, max_size)
            
            # base64 데이터로 비전 메시지 생성
            return self.create_vision_message(text=text, image_base64=image_base64, use_data_uri=use_data_uri)
            
        except Exception as e:
            raise Exception(f"웹 이미지 다운로드 및 메시지 생성 실패: {str(e)}")

    def analyze_web_image_with_download(self, image_url, question="이 이미지에 무엇이 보이나요? 자세히 설명해주세요.", 
                                        system_prompt=None, verbose=True, max_size=(1024, 1024), use_data_uri=True, **kwargs):
        """
        웹 URL 이미지를 다운로드받아 분석하는 완전한 워크플로우
        (URL 직접 전송이 아닌 다운로드 후 파일 데이터로 전송)
        
        Args:
            image_url (str): 분석할 이미지 URL
            question (str): 이미지에 대한 질문
            system_prompt (str, optional): 시스템 프롬프트
            verbose (bool): 진행 상황 출력 여부
            max_size (tuple): 이미지 최대 크기
            use_data_uri (bool): True시 dataUri 방식, False시 imageUrl 방식으로 base64 전송
            **kwargs: API 파라미터들
        
        Returns:
            dict: API 응답 결과
        """
        try:
            if verbose:
                print("🔍 웹 이미지 다운로드 및 분석 시작...")
                print(f"📎 이미지 URL: {image_url}")
                print("⬇️ 이미지 다운로드 중...")
                print(f"📋 전송 방식: {'dataUri' if use_data_uri else 'imageUrl'} (base64)")
            
            # 기본 시스템 프롬프트
            if system_prompt is None:
                system_prompt = "당신은 이미지를 자세히 분석하고 설명하는 전문 AI 어시스턴트입니다. 한국어로 정확하고 자세하게 답변해주세요."
            
            # 웹 이미지를 다운로드하여 비전 메시지 생성
            vision_message = self.create_vision_message_with_downloaded_image(
                text=question, 
                image_url=image_url,
                max_size=max_size,
                use_data_uri=use_data_uri
            )
            
            # 시스템 메시지를 올바른 형식으로 구성
            system_message = {
                "role": "system", 
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            }
            
            messages = [
                system_message,
                vision_message
            ]
            
            if verbose:
                print("✅ 이미지 다운로드 완료!")
                print("🤖 AI 모델 분석 중...")
            
            # API 호출
            result = self.chat_completions_v3(
                messages=messages,
                model_name="HCX-005",
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000),
                top_p=kwargs.get("top_p", 0.8),
                top_k=kwargs.get("top_k", 0),
                repetition_penalty=kwargs.get("repetition_penalty", 1.1),
                stop=kwargs.get("stop", [])
            )
            
            if verbose:
                if "error" not in result:
                    print("✅ 분석 완료!")
                else:
                    print(f"❌ 분석 실패: {result.get('error')}")
            
            return result
            
        except Exception as e:
            error_result = {"error": f"이미지 다운로드 및 분석 실패: {str(e)}"}
            if verbose:
                print(f"❌ 오류 발생: {str(e)}")
            return error_result


def main():
    """사용 예제 - cURL 방식과 동일한 구조"""
    # API 키 설정
    CLOVA_API_KEY = NAVER_CLOVA_API_KEY  # 클로바 스튜디오 API 키
    
    # 클로버 스튜디오 API 인스턴스 생성
    clova = ClovaStudioAPI(CLOVA_API_KEY)
    
    print("=" * 80)
    print("🚀 Clova Studio API 웹 이미지 분석 테스트 (cURL 호환)")
    print("=" * 80)
    
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
    
    # 테스트 이미지 URL
    test_image_url = "https://web-front.callmaner.com/new_web_s/image/i_7e65caaab476.jpg"

    # === 1. cURL 예제와 동일한 구조로 이미지 분석 ===
    print("\n1️⃣ cURL 방식 이미지 분석")
    print("-" * 50)
    
    # 사용자 메시지 생성 (cURL 예제와 동일한 구조)
    user_message = clova.create_vision_message(
        text="이 이미지에 포함된 모든 텍스트에서 보내신분, 품명, 수량, 경.조사어, 배달일시, 배달장소, 기타, 받으신분, 인수하신분, 서명까지 모두 정확히 추출해주세요. 전화번호도 있으면 표시해주세요. 텍스트가 없다면 '텍스트 없음'이라고 답변해주세요.",
#        text="이 이미지에 포함된 모든 정보를 추출합니다. 추출된 정보를 목록 형태로 정리합니다.",
        image_url=test_image_url
    )
    
    messages = [system_message, user_message]

    print(messages)
    
    result = clova.chat_completions_v3(
        messages=messages,
        model_name="HCX-005",
        top_p=0.8,
        top_k=0,
        max_tokens=100,
        temperature=0.5,
        repetition_penalty=1.1,
        stop=[]
    )
    
    if "error" not in result:
        print("✅ 분석 결과:")
        if "result" in result and "message" in result["result"]:
            print(result["result"]["message"]["content"])
        elif "choices" in result and len(result["choices"]) > 0:
            print(result["choices"][0]["message"]["content"])
        else:
            print("응답 형식을 확인할 수 없습니다:", result)
    else:
        print(f"❌ 실패: {result.get('error')}")

""" 
    print("\n1️⃣ dataUri 방식으로 웹 이미지 다운로드 후 분석")
    print("-" * 50)
    
    # dataUri 방식 (순수 base64 데이터만 전송)
    try:
        user_message = clova.create_vision_message_with_downloaded_image(
            text="이 이미지의 모든 내용을 추출해서 정리해줘",
            image_url=test_image_url,
            max_size=(1024, 1024),
            use_data_uri=True  # dataUri 방식 사용
        )
        
        messages = [system_message, user_message]
        print("📋 전송 방식: dataUri (순수 base64)")
        print("📦 메시지 구조:", json.dumps(user_message, indent=2, ensure_ascii=False)[:200] + "...")
        
        result = clova.chat_completions_v3(
            messages=messages,
            model_name="HCX-005",
            top_p=0.8,
            top_k=0,
            max_tokens=1000,
            temperature=0.5,
            repetition_penalty=1.1,
            stop=[]
        )
        
        if "error" not in result:
            print("✅ dataUri 방식 분석 결과:")
            if "result" in result and "message" in result["result"]:
                print(result["result"]["message"]["content"])
            elif "choices" in result and len(result["choices"]) > 0:
                print(result["choices"][0]["message"]["content"])
            else:
                print("응답 형식을 확인할 수 없습니다:", result)
        else:
            print(f"❌ dataUri 방식 실패: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ dataUri 방식 오류 발생: {str(e)}")

    print("\n" + "=" * 80)
    print("🎉 모든 테스트 완료!")
    print("=" * 80)        

"""

if __name__ == "__main__":
    main()