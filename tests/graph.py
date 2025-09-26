# graph.py

# ========== 임포트 ==========
from langgraph.graph import StateGraph, START, END
from state import State
from nodes import refine_question, retrieve, rerank, generate_rag_answer, verify_rag_answer, generate_contact_answer
from router import update_hr_status, route_after_hr, generate_reject_answer, route_after_rag, update_rag_status


# ========== 그래프 빌더 ==========
builder = StateGraph(State)

# ========== 노드 등록(흐름 순서) ==========
# 흐름: START -> refine_question -> hr_node -> (router2 | reject)
# 흐름: router2 -> (retrieve | department)
# 흐름: retrieve -> rerank -> generate_rag_answer -> verify_rag_answer -> END

# 사전 쿼리 분석
builder.add_node("refine_question", refine_question)

# 1차 라우터
builder.add_node("update_hr_status", update_hr_status)
builder.add_node("generate_reject_answer", generate_reject_answer)  # 터미널

# 2차 라우터 및 담당자 안내
builder.add_node("update_rag_status", update_rag_status)
builder.add_node("generate_contact_answer", generate_contact_answer)  # 터미널

# RAG 파이프라인
builder.add_node("retrieve", retrieve)
builder.add_node("rerank", rerank)
builder.add_node("generate_rag_answer", generate_rag_answer)
builder.add_node("verify_rag_answer", verify_rag_answer)

# ========== 엣지(흐름 순서) ==========
# 시작과 쿼리 분석
builder.add_edge(START, "refine_question")
builder.add_edge("refine_question", "update_hr_status")

# 1차 라우터: HR이면 router2, 아니면 reject
builder.add_conditional_edges(
    "update_hr_status",
    route_after_hr,
    {"router2": "update_rag_status", "reject": "generate_reject_answer"},
)

# 2차 라우터: rag는 retrieve(검색), department는 터미널
builder.add_conditional_edges(
    "update_rag_status",
    route_after_rag,
    {"rag": "retrieve", "department": "generate_contact_answer"},
)

# RAG 파이프라인
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate_rag_answer")
builder.add_edge("generate_rag_answer", "verify_rag_answer")
builder.add_edge("verify_rag_answer", END)

# 터미널 경로
builder.add_edge("generate_contact_answer", END)
builder.add_edge("generate_reject_answer", END)

# ========== 공개 그래프 ==========
graph = builder.compile()