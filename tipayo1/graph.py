# graph.py

# ========== 임포트 ==========
from langgraph.graph import StateGraph, START, END
from state import State
from nodes import analyze_query, retrieve, rerank, generate_answer, verify_answer
from router import (
    hr_node,
    route_after_hr,
    reject_node,
    router2_node,
    route_after_router2,
    department_node,
)

# ========== 그래프 빌더 ==========
builder = StateGraph(State)

# ========== 노드 등록(흐름 순서) ==========
# 흐름: START -> analyze_query -> hr_node -> (router2 | reject)
# 흐름: router2 -> (retrieve | department)
# 흐름: retrieve -> rerank -> generate_answer -> verify_answer -> END

# 사전 쿼리 분석
builder.add_node("analyze_query", analyze_query)

# 1차 라우터
builder.add_node("hr_node", hr_node)
builder.add_node("reject", reject_node)  # 터미널

# 2차 라우터 및 담당자 안내
builder.add_node("router2", router2_node)
builder.add_node("department", department_node)  # 터미널

# RAG 파이프라인
builder.add_node("retrieve", retrieve)
builder.add_node("rerank", rerank)
builder.add_node("generate_answer", generate_answer)
builder.add_node("verify_answer", verify_answer)

# ========== 엣지(흐름 순서) ==========
# 시작과 쿼리 분석
builder.add_edge(START, "analyze_query")
builder.add_edge("analyze_query", "hr_node")

# 1차 라우터: HR이면 router2, 아니면 reject
builder.add_conditional_edges(
    "hr_node",
    route_after_hr,
    {"router2": "router2", "reject": "reject"},
)

# 2차 라우터: rag는 retrieve(검색), department는 터미널
builder.add_conditional_edges(
    "router2",
    route_after_router2,
    {"rag": "retrieve", "department": "department"},
)

# RAG 파이프라인
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate_answer")
builder.add_edge("generate_answer", "verify_answer")
builder.add_edge("verify_answer", END)

# 터미널 경로
builder.add_edge("department", END)
builder.add_edge("reject", END)

# ========== 공개 그래프 ==========
graph = builder.compile()

# ========== 로컬 테스트 진입점(선택) ==========
if __name__ == "__main__":
    from langgraph.checkpoint.memory import InMemorySaver

    demo_graph = builder.compile(checkpointer=InMemorySaver())

    init_state: State = {
        "messages": [],
        "question": "연차휴가는 언제까지 사용해야 하나요?",
        # "category": "연차휴가",
    }
    config = {"configurable": {"thread_id": "demo-thread-1"}}
    out_state = demo_graph.invoke(init_state, config=config)
    print("최종 답변:\n", out_state.get("result", ""))
    print("검증 결과:\n", out_state.get("verification", ""))
