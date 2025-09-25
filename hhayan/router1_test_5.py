from typing_extensions import Literal, TypedDict
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from typing import cast

# router1: - HR 질문이면 → "router2" else "reject"
# router1은 단순히 “HR 여부를 판별해서 state에 기록하고, 다음 방향 정하기, 노드가 아님

# =========================
# 1. State 정의
# =========================
class State(MessagesState, total=False):
    refined_question: str
    is_hr_question: bool
    answer_type: Literal["pending", "reject"]

# =========================
# 2. HR Router (라우터 역할)
# =========================
class HRAnalysis(TypedDict):
    is_hr_question: bool

def hr_router(state: State) -> str:
    """
    HR 여부만 LLM으로 판별합니다.
    판별 결과를 state에 기록하고, 라우팅 방향("router2" or "reject")은 Python에서 반환합니다.
    """
    prompt = f"""
    당신은 "가이다 플레이 스튜디오(GPS)"의 HR 정책 안내 챗봇입니다.
    아래 질문이 HR(인사/근무/휴가/복지/장비·보안/출장·비용처리 등) 관련인지 판별하세요.

    질문: "{state['refined_question']}"
    """

    structured_llm = llm.with_structured_output(HRAnalysis)
    result: HRAnalysis = structured_llm.invoke(prompt) # dict 형태: {"is_hr_question": True or false}

    # state 업데이트
    state["is_hr_question"] = result["is_hr_question"]
    state["answer_type"] = "pending" if result["is_hr_question"] else "reject"

    # 라우팅 방향만 반환
    return "router2" if result["is_hr_question"] else "reject" # ture -> router2 / false -> reject
# =========================
# 3. Reject Node
# =========================
def reject_node(state: State):
    """HR 관련이 아닌 질문에 대한 거부 메시지"""
    return "입력하신 질문은 HR 관련 문의가 아닙니다. HR 관련 질문만 가능합니다."

