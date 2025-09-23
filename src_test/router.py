# router.py

from typing import Dict, List, Optional, TypedDict, Literal, cast
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import State


# =========================
# 1차 라우터: HR Router
# =========================
class HRAnalysis(TypedDict):
    is_hr_question: bool

def hr_router(state: State) -> State:
    """
    HR 여부만 판별, 그 결과를 상태에 저장
    """
    prompt = f"""

    질문: "{state['refined_question']}"

    당신은 "가이다 플레이 스튜디오(GPS)"의 HR 정책 안내 챗봇입니다.
    아래 질문이 HR(인사/근무/휴가/복지/장비·보안/출장·비용처리 등) 관련인지 판별하세요.
    """
    _llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    structured_llm = _llm.with_structured_output(HRAnalysis)

    result: HRAnalysis = structured_llm.invoke(prompt)
    is_hr = result["is_hr_question"]

    # HR 여부에 따라 answer_type 세팅
    answer_type = "pending" if is_hr else "reject"

    return cast(State, {**state, "is_hr_question": is_hr, "answer_type": answer_type})

# 라우터 함수. Return 은 다음으로 이어질 Node 의 이름
def route_after_hr(state: State) -> str:
    """HR 판별 결과에 따라 다음 노드 결정"""
    # HR 질문이면 router2로, 아니면 reject로
    return "router2" if state["is_hr_question"] else "reject"

# =========================
# Answer_type: Reject
# =========================
def reject_node(state: State):
    """HR 관련이 아닌 질문에 대한 거부 메시지"""
    return "입력하신 질문은 HR 관련 문의가 아닙니다. HR 관련 질문만 가능합니다."


