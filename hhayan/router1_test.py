from dotenv import load_dotenv
from pprint import pprint
from typing_extensions import Literal

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

# =========================
# 0. 환경변수 로드
# =========================
load_dotenv()

# LLM (백업용, 규칙 기반에서 못 잡을 때만 사용)
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# =========================
# 1. State 정의
# =========================
from typing import TypedDict
class Router1State(TypedDict):
    question: str
    is_hr_question: Literal["none", "work", "leave", "equipment", "welfare", "etc"]
    next_step: Literal["rag", "reject"]
    answer: str | None
# =========================
# 2. Router1 Node
# =========================
def router1_node(state: Router1State) -> Router1State:
    question = state["question"].lower()

    hr_keywords = {
        "work": ["근무", "출근", "퇴근", "재택", "야근", "코어타임"],
        "leave": ["연차", "휴가", "반차", "병가", "경조사"],
        "equipment": ["노트북", "장비", "보안", "카드키", "인증"],
        "welfare": ["복지", "포인트", "식대", "교육비", "상담"],
        "etc": ["급여", "월급", "비용", "출장", "정산", "식비"],
    }

    # 1차 규칙 기반 분류
    is_hr = "none"
    for category, keywords in hr_keywords.items():
        if any(kw in question for kw in keywords):
            is_hr = category
            break

    # 2차: 규칙 기반에서 못 잡으면 LLM 보정
    if is_hr == "none":
        prompt = f"""
        당신은 HR 분류 어시스턴트입니다.
        다음 질문이 HR과 관련 있는지 판단하세요.
        카테고리: work(근무), leave(휴가), equipment(장비보안), welfare(복지), etc(급여·비용·출장), none(HR 무관)

        출력은 아래 형식으로만 하세요:
        is_hr_question: [카테고리]
        next_step: rag | reject

        질문: "{state['question']}"
        """
        response = llm.invoke(prompt).content

        if "work" in response: is_hr = "work"
        elif "leave" in response: is_hr = "leave"
        elif "equipment" in response: is_hr = "equipment"
        elif "welfare" in response: is_hr = "welfare"
        elif "etc" in response: is_hr = "etc"
        else: is_hr = "none"

    next_step = "rag" if is_hr != "none" else "reject"

    return {**state, "is_hr_question": is_hr, "next_step": next_step}

# =========================
# 3. Router
# =========================
def route_after_router1(state: Router1State) -> str:
    return "reject" if state["next_step"] == "reject" else "router2"

# =========================
# 4. Reject Node
# =========================
def reject_node(state: Router1State) -> dict:
    return {**state, "answer": "❌ 지원하지 않는 질문입니다. HR 정책 관련 문의만 가능합니다."}

# =========================
# 5. 그래프 빌드
# =========================
builder = StateGraph(Router1State)
builder.add_node("router1", router1_node)
builder.add_node("reject", reject_node)

builder.add_conditional_edges(
    "router1",
    route_after_router1,
    {"router2": END, "reject": "reject"}  # router2는 테스트용이므로 END로 연결
)

builder.add_edge("reject", END)
builder.add_edge(START, "router1")

graph = builder.compile()
