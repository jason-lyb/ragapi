import aiohttp
import base64
import os
from PIL import Image, ExifTags
from urllib.parse import urlparse
from logger import logger
from config import TEMP_DIR

# 임시 디렉토리 생성
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

async def fetch_image_metadata(request_id, url):
    """이미지 URL에서 메타데이터를 가져옵니다."""
    logger.debug(f"[{request_id}] 이미지 메타데이터 가져오기: {url}")
    
    try:
        # URL에서 정보 추출
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path) or "image_from_url"
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # 헤더 정보 확인 (선택 사항)
        img_format = 'unknown'
        file_size = None

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

async def download_image(request_id, url):
    """URL에서 이미지를 다운로드하고 base64로 인코딩합니다."""
    logger.debug(f"[{request_id}] 이미지 다운로드 시작: {url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    logger.error(f"[{request_id}] 이미지 다운로드 실패: 상태 코드 {response.status}")
                    raise Exception(f"이미지 다운로드 실패: 상태 코드 {response.status}")
                
                image_data = await response.read()
                img_base64 = base64.b64encode(image_data).decode('utf-8')
                
                logger.debug(f"[{request_id}] 이미지 다운로드 완료: {len(image_data)} 바이트")
                
                return img_base64, image_data
                
    except Exception as e:
        logger.error(f"[{request_id}] 이미지 다운로드 중 오류: {e}")
        raise Exception(f"이미지 다운로드 중 오류: {str(e)}")

async def extract_image_metadata(request_id, image_data, file_name):
    """이미지 데이터에서 메타데이터를 추출합니다."""
    logger.debug(f"[{request_id}] 이미지 메타데이터 추출 시작")
    
    try:
        # 임시 파일 경로
        temp_file_path = os.path.join(TEMP_DIR, f"temp_{request_id}_{file_name}")
        
        # 이미지 데이터를 임시 파일로 저장
        with open(temp_file_path, "wb") as buffer:
            buffer.write(image_data)
        
        # PIL을 사용하여 이미지 메타데이터 추출
        with Image.open(temp_file_path) as img:
            metadata = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode
            }
            
            # EXIF 데이터 추출 시도
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif():
                for tag, value in img._getexif().items():
                    if tag in ExifTags.TAGS:
                        exif_data[ExifTags.TAGS[tag]] = str(value)
            
            # 중요 EXIF 정보만 선택
            important_exif = {}
            for key in ['DateTime', 'Make', 'Model', 'GPSInfo', 'ExposureTime', 'FNumber', 'ISOSpeedRatings']:
                if key in exif_data:
                    important_exif[key] = exif_data[key]
                    
            metadata["exif"] = important_exif
        
        # 임시 파일 삭제
        os.remove(temp_file_path)
        
        return metadata
        
    except Exception as e:
        logger.warning(f"[{request_id}] 이미지 메타데이터 추출 중 오류: {e}")
        # 임시 파일이 존재하면 삭제
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return {
            "error": f"메타데이터 추출 실패: {str(e)}"
        }