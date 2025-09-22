# Graph 생성, 노드 연결, 파이프라인 실행 예시를 포함합니다.

# graph.py

from langgraph.graph import Graph

from state import State

from node import (
    setup_vectorstore,
    analyze_query,
    retrieve,
    rerank,
    generate_answer,
    verify_answer,
    start,
    end
)

# Graph 구성
graph = Graph(State)

# 노드 추가
graph.add_node(start, requires=[])
graph.add_node(setup_vectorstore, requires=[])
graph.add_node(analyze_query, requires=['question'])
graph.add_node(retrieve, requires=['query', 'vectorstore'])
graph.add_node(rerank, requires=['docs'])
graph.add_node(generate_answer, requires=['question', 'docs'])
graph.add_node(verify_answer, requires=['result', 'docs'])
graph.add_node(end, requires=[])

# 엣지 구성 (직렬 흐름)
graph.add_edge(start, setup_vectorstore)
graph.add_edge(setup_vectorstore, analyze_query)
graph.add_edge(analyze_query, retrieve)
graph.add_edge(retrieve, rerank)
graph.add_edge(rerank, generate_answer)
graph.add_edge(generate_answer, verify_answer)
graph.add_edge(verify_answer, end)

# 그래프 컴파일
graph.compile()

# 실행 예시
if __name__ == "__main__":
    # 1) 초기 state 생성
    state = State(question="연차휴가는 언제까지 사용해야 하나요?")

    # 2) start 노드 실행으로 초기 입력 처리
    state = start(state)

    # 3) vectorstore 설정 후 state에 주입
    setup_out = setup_vectorstore(state)
    state.vectorstore = setup_out["vectorstore"]

    # 4) 파이프라인 실행
    state = graph.run(state)

    # 5) end 노드 실행 (후처리, 없는 경우 생략 가능)
    state = end(state)

    # 6) 결과 출력
    print("최종 답변:\n", state.result)
    print("검증 결과:\n", state.verification)
