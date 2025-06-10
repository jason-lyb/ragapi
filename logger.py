import logging
import sys
from logging.handlers import RotatingFileHandler
import os
import datetime

# 로그 디렉토리 생성
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 오늘 날짜를 기반으로 로그 파일명 생성
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_DIR, f"rag_api_{current_date}.log")

# 로거 설정
logger = logging.getLogger("rag_api")
logger.setLevel(logging.DEBUG)  # 루트 로거를 DEBUG로 변경

# 중복 핸들러 방지
if not logger.handlers:
    # 파일 핸들러 설정 (DEBUG 레벨로 모든 로그 저장)
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러 설정 (INFO 레벨로 중요한 로그만 출력)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# 상위 로거로의 전파 방지 (선택적)
logger.propagate = False