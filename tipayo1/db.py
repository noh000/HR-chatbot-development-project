# 환경변수 로드, 로깅 및 Pinecone 초기화를 담당합니다.

# db.py
from dotenv import load_dotenv
import os
import logging
import pinecone

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# Pinecone 초기화
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
)
