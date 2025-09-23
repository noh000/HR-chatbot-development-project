# router1 node: HR NODE + REJECT NODE

# =========================
# 2. HR Node 
# =========================

# json형식 확정용
class HRAnalysis(TypedDict):
    """HR 여부 판별 결과"""
    is_hr_question: bool                  # True → HR 관련, False → HR 아님
    next_step: str                        # "router2" | "reject"

def hr_node(state: State) -> State:
    """HR 관련 질문인지 판별하는 노드"""
    prompt = f"""
    당신은 "가이다 플레이 스튜디오(GPS)"의 HR 정책 안내 챗봇입니다.  
    회사 내부 직원의 질문이 HR(인사/근무/휴가/복지/장비·보안/출장·비용처리 등)과 관련된지 판별하세요.  

    질문: "{state['refined_question']}"
    """

    structured_llm = llm.with_structured_output(HRAnalysis)

    try:
        result: HRAnalysis = structured_llm.invoke(prompt)
    except Exception:
        # fallback: HR 아님 처리
        return {**state, "is_hr_question": False, "next_step": "reject"}

    return {
        **state,
        "is_hr_question": result["is_hr_question"],
        "next_step": result["next_step"],
    }

# =========================
# 5. Reject Node
# =========================
def reject_node(state: State):
    """HR 관련이 아닌 질문에 대한 거부 메시지 후 종료"""
    return "입력하신 질문은 HR 관련 문의가 아닙니다. HR 관련 질문만 가능합니다."