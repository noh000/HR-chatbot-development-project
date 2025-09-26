# state.py
from typing import Literal, Optional, Dict, List
from langgraph.graph import MessagesState
from langchain_core.documents import Document

class State(MessagesState, total=False):
    """HR 챗봇 메인 상태 클래스"""
    
    # === 사용자 질문 전처리 ===
    user_question: str                          # 사용자가 입력한 원본 질문
    refined_question: str                       # LLM이 정제한 질문 (문맥 보완, 맞춤법 교정 등)

    # === 1차 라우터: HR 관련 질문 판단 ===
    is_hr_question: bool                        # True: HR 관련, False: 비관련 질문
    
    # === 2차 라우터: 답변 방식 결정 ===
    is_rag_suitable: bool                       # True: RAG 검색 가능, False: 담당자 연결 필요
    
    # === 담당자 안내 정보 ===
    department_info: Optional[Dict[str, str]]   # 담당 부서 연락처 {"name": "부서명", "email": "이메일", "phone": "전화번호", "slack": "슬랙채널"}
    
    # === RAG 처리 ===
    retrieved_docs: List[Document]              # 벡터DB에서 검색된 관련 문서들 (Top-K)
        
    # === 최종 답변 통합 관리 ===
    answer_type: Literal[                       # 답변 유형 구분
        "pending",                              # 1차 라우터 True
        "reject",                               # 1차 라우터 False
        "rag_answer",                           # 2차 라우터 True
        "department_contact"                    # 2차 라우터 False
    ]
    final_answer: str                           # 사용자에게 제공할 최종 답변 (모든 유형 통합)
    
    # === 처리 상태 추적 ===
    # current_step: Literal[                      # 현재 처리 단계 추적
    #     "preprocess",                           # 질문 전처리 중
    #     "hr_routing",                           # 1차 라우팅 중  
    #     "answer_routing",                       # 2차 라우팅 중
    #     "rag_processing",                       # RAG 처리 중
    #     "generating_answer",                    # 답변 생성 중
    #     "completed"                             # 처리 완료
    # ]

    # === MVP 개발 후 고려 ===
    # category: str                           # 전처리 노드에서 분류된 카테고리
    # reranked_docs: List[Document]           # 검색 후 재순위화된 Document 리스트
    verification: str                       # 답변 품질 검증 결과