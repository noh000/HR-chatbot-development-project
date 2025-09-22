from dotenv import load_dotenv
from typing_extensions import Literal, TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI

# =========================
# 0. 환경 변수 로드 & LLM
# =========================
load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# =========================
# 1. State 정의 (query 제거)
# =========================
class State(MessagesState, total=False):
    question: str
    is_hr_question: bool
    next_step: Literal["router2", "reject"]
    answer: str

# =========================
# 2. HR Node
# =========================
class HRAnalysis(TypedDict):
    is_hr_question: bool
    next_step: Literal["router2", "reject"]

def hr_node(state: State) -> State:
    """HR 관련 질문인지 판별하는 노드"""
    prompt = f"""
    당신은 "가이다 플레이 스튜디오(GPS)"의 HR 정책 안내 챗봇입니다.
    회사 내부 직원의 질문이 HR(인사/근무/휴가/복지/장비·보안/출장·비용처리 등)과 관련된지 판별하세요.

    질문: "{state.get('question', '')}"
    """

    structured_llm = llm.with_structured_output(HRAnalysis)

    try:
        result: HRAnalysis = structured_llm.invoke(prompt)
    except Exception:
        # fallback: HR 아님 처리
        return {
            **state,
            "is_hr_question": False,
            "next_step": "reject"
        }

    return {
        **state,
        "is_hr_question": result.get("is_hr_question", False),  # 항상 기본값 보장
        "next_step": result.get("next_step", "reject"),
    }

# =========================
# 3. HR Router
# =========================
def route_after_hr(state: State) -> str:
    """HR 판별 결과에 따라 다음 노드 결정"""
    is_hr = state.get("is_hr_question", False)  # 기본값 False
    return "router2" if is_hr else "reject"


# =========================
# 4. Reject Node
# =========================
def reject_node(state: State) -> State:
    return {**state, "answer": "HR 관련 문의만 가능합니다."}

# =========================
# 5. Router2 Node
# =========================
def router2_node(state: State) -> State:
    return {**state, "answer": f"HR 질문: {state['question']}"}

# =========================
# 6. 그래프 정의
# =========================
def build_graph():
    graph = StateGraph(State)

    graph.add_node("hr_node", hr_node)
    graph.add_node("reject", reject_node)
    graph.add_node("router2", router2_node)

    graph.add_edge(START, "hr_node")
    graph.add_conditional_edges("hr_node", route_after_hr, {"router2": "router2", "reject": "reject"})
    graph.add_edge("reject", END)
    graph.add_edge("router2", END)

    return graph.compile()

# =========================
# 7. LangGraph CLI용 Export
# =========================
graph = build_graph()
