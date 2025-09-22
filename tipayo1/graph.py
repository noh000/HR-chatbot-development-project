# graph.py
from langgraph.graph import StateGraph, START, END
from state import State
from node import analyze_query, retrieve, rerank, generate_answer, verify_answer

# Graph 구성
builder = StateGraph(State)

# 노드
builder.add_node("analyze_query", analyze_query)
builder.add_node("retrieve", retrieve)
builder.add_node("rerank", rerank)
builder.add_node("generate_answer", generate_answer)
builder.add_node("verify_answer", verify_answer)

# 엣지
builder.add_edge(START, "analyze_query")
builder.add_edge("analyze_query", "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate_answer")
builder.add_edge("generate_answer", "verify_answer")
builder.add_edge("verify_answer", END)

# LangGraph API에 노출되는 그래프: 사용자 정의 체크포인터 없이 컴파일
graph = builder.compile()

if __name__ == "__main__":
    # 선택: 로컬 단독 실행 시에만 InMemorySaver로 디버깅
    from langgraph.checkpoint.memory import InMemorySaver
    demo_graph = builder.compile(checkpointer=InMemorySaver())

    init_state: State = {
        "messages": [],
        "question": "연차휴가는 언제까지 사용해야 하나요?",
    }
    config = {"configurable": {"thread_id": "demo-thread-1"}}
    out_state = demo_graph.invoke(init_state, config=config)

    print("최종 답변:\n", out_state.get("result", ""))
    print("검증 결과:\n", out_state.get("verification", ""))
