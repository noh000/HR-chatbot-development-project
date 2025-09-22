# state.py
from typing import List
from langgraph.graph import MessagesState
from langchain_core.documents import Document

class State(MessagesState, total=False):
    """
    - question: 사용자가 입력한 원문 질문
    - query: 벡터 검색용 쿼리
    - docs: 검색 후 재순위화된 Document 리스트
    - result: 최종 답변
    - verification: 답변 검증 결과
    """
    question: str
    query: str
    docs: List[Document]
    result: str
    verification: str
