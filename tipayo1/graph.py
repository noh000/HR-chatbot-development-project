# graph.py

from langgraph.graph import StateGraph, START, END
from state import State
from nodes import analyze_query, retrieve, rerank, generate_answer, verify_answer
from router import (
    hr_node, route_after_hr, reject_node,
    router2_node, route_after_router2, department_node
)

# Graph 구성
builder = StateGraph(State)

# 1차 라우터 노드
builder.add_node("hr_node", hr_node)
builder.add_node("reject", reject_node)

# 2차 라우터 및 담당자 안내 노드
builder.add_node("router2", router2_node)
builder.add_node("department", department_node)

# RAG 파이프라인 노드
builder.add_node("analyze_query", analyze_query)
builder.add_node("retrieve", retrieve)
builder.add_node("rerank", rerank)
builder.add_node("generate_answer", generate_answer)
builder.add_node("verify_answer", verify_answer)

# 엣지 (요청에 따른 재배치)
builder.add_edge(START, "analyze_query")
builder.add_edge("analyze_query", "hr_node")

# 1차 라우터 분기: HR이면 router2, 아니면 reject
builder.add_conditional_edges(
    "hr_node",
    route_after_hr,
    {
        "router2": "router2",
        "reject": "reject",
    },
)

# 2차 라우터 분기: rag -> RAG 파이프라인 시작(retrieve), department -> 담당자 안내(터미널)
builder.add_conditional_edges(
    "router2",
    route_after_router2,
    {
        "rag": "retrieve",
        "department": "department",
    },
)

# RAG 파이프라인
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate_answer")
builder.add_edge("generate_answer", "verify_answer")
builder.add_edge("verify_answer", END)

# 터미널들
builder.add_edge("department", END)
builder.add_edge("reject", END)

# LangGraph API에 노출되는 그래프
graph = builder.compile()

if __name__ == "__main__":
    # 선택: 로컬 단독 실행 시에만 InMemorySaver로 디버깅
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
