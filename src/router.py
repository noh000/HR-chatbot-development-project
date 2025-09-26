# router.py

from typing import Literal
from state import State


# =========================
# 1차 라우터: HR Router
# =========================

def route_after_hr(state: State) -> str:
    """HR 판별 결과에 따라 다음 노드 결정"""
    # HR 질문이면 router2로, 아니면 reject로
    return "router2" if state["is_hr_question"] else "reject"


# =========================
# 2차 라우터: RAG vs Department
# =========================

def route_after_rag(state: State) -> Literal["rag", "department"]:
    """RAG 사용 여부에 따라 다음 노드 결정"""
    if state.get('is_rag_suitable'):
        return "rag"
    else:
        return "department"