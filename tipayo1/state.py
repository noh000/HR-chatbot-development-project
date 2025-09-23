# state.py

from typing import List, Optional, Dict
from typing_extensions import Literal
from langgraph.graph import MessagesState
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import os

class State(MessagesState, total=False):
    """
    - question: 사용자가 입력한 원문 질문
    - category: 사전 전처리된 카테고리(없으면 미분류로 동작)
    - is_hr_question: HR 여부(라벨)
    - next_step: HR 라우터의 다음 분기 키(router2 또는 reject)
    - query: 벡터 검색용 쿼리
    - docs: 검색 후 재순위화된 Document 리스트
    - result: 최종 답변
    - verification: 답변 검증 결과
    - is_rag: 2차 라우터 결과 (True면 RAG 경로)
    - department: 2차 라우터 결과 부서 정보 (route=department일 때)
    """
    question: str
    category: str
    is_hr_question: Literal["yes", "no"]
    next_step: Literal["router2", "reject"]
    query: str
    docs: List[Document]
    result: str
    verification: str

    # 2차 라우터용 확장 필드
    is_rag: bool
    department: Optional[Dict[str, str]]

# 공용 LLM 팩토리: 역할별 모델 통일
def get_llm(role: str = "gen") -> ChatOpenAI:
    """
    role:
    - "gen": 본문 생성/분석 (기본 gpt-4.1)
    - "router": 2차 라우터 (기본 gpt-4.1-mini)
    - "router-lite": 1차 HR 분류 (기본 gpt-4.1-nano)
    환경변수로 덮어쓰기:
    GEN_LLM, ROUTER_LLM, ROUTER_LITE_LLM
    """
    model_map = {
        "gen": os.getenv("GEN_LLM", "gpt-4.1"),
        "router": os.getenv("ROUTER_LLM", "gpt-4.1-mini"),
        "router-lite": os.getenv("ROUTER_LITE_LLM", "gpt-4.1-nano"),
    }
    model = model_map.get(role, model_map["gen"])
    return ChatOpenAI(model=model, temperature=0)
