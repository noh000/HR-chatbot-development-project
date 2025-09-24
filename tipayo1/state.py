# state.py
from typing import Literal, Optional, Dict, List
from langgraph.graph import MessagesState
from langchain_core.documents import Document

class State(MessagesState, total=False):
    """HR 챗봇 메인 상태 클래스"""

    # 사용자 질문 전처리
    user_question: str
    refined_question: str

    # 1차 라우터
    is_hr_question: bool

    # 2차 라우터
    is_rag_suitable: bool

    # 담당자 안내 정보
    department_info: Optional[Dict[str, str]]

    # RAG 처리
    retrieved_docs: List[Document]

    # 최종 답변 통합 관리
    answer_type: Literal["pending", "reject", "rag_answer", "department_contact"]
    final_answer: str

    # 선택적: rerank 결과/검증 결과 등을 추후 확장 가능
    # verification: str
