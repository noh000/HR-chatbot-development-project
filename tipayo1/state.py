# LangGraph MessagesState를 상속한 상태 정의만 포함합니다.

# state.py
from typing import List
from langgraph.graph import MessagesState
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

class State(MessagesState):
    """
    MessagesState 상속

    - question: 사용자가 입력한 원문 질문
    - query: 벡터 검색용 쿼리
    - docs: 검색 후 재순위화된 Document 리스트
    - result: 최종 답변
    - verification: 답변 검증 결과
    - vectorstore: 외부에서 주입되는 PineconeVectorStore 객체
    """
    question: str
    query: str
    docs: List[Document]
    result: str
    verification: str
    vectorstore: PineconeVectorStore
