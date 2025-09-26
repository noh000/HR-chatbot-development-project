from typing_extensions import Literal, TypedDict
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI

# router1: - HR 질문이면 → "router2" else "reject"
# router1은 단순히 “HR 여부를 판별해서 state에 기록” 까지만
# 라우팅 결정은 LangGraph 연결 단계에서 처리

# =========================
# 1. State 정의
# =========================
class State(MessagesState, total=False):
    refined_question: str
    is_hr_question: bool
    answer_type: Literal[
        "pending",   # 1차 라우터 통과 (HR 관련 질문 → router2로 진행)
        "reject"     # 1차 라우터에서 걸러짐 (HR 무관 질문)
    ]

# =========================
# 2. HR Router (1차 라우터)
# =========================

# is_hr_question의 output 형태 확정 
class HRAnalysis(TypedDict):
    is_hr_question: bool

def hr_router(state: State) -> State:
    """
    HR 여부만 판별, 그 결과를 상태에 저장
    """
    prompt = f"""
    당신은 "가이다 플레이 스튜디오(GPS)"의 HR 정책 안내 챗봇입니다.
    아래 질문이 HR(인사/근무/휴가/복지/장비·보안/출장·비용처리 등) 관련인지 판별하세요.

    질문: "{state['refined_question']}"

    """
    
    structured_llm = llm.with_structured_output(HRAnalysis)

    try:
        result: HRAnalysis = structured_llm.invoke(prompt)
        is_hr = result["is_hr_question"]

    # 예외 처리 단순화: 지금은 실패 시 False로 처리하는데, 필요하다면 "error" 같은 상태를 별도로 두는 것도 고려 가능 (answer_type = "reject_error" 등).    
    except Exception:
        is_hr = False

    # HR 여부에 따라 answer_type 세팅
    answer_type = "pending" if is_hr else "reject"

    return {**state, "is_hr_question": is_hr, "answer_type": answer_type}

# =========================
# 3. Reject Node
# =========================
def reject_node(state: State):
    """HR 관련이 아닌 질문에 대한 거부 메시지"""
    return "입력하신 질문은 HR 관련 문의가 아닙니다. HR 관련 질문만 가능합니다."

