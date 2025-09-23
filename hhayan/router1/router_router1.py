# router1 router

# =========================
# 2. HR Router (1차 라우터)
# =========================
from typing import cast

# is_hr_question의 출력 스키마 강제 지정
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

    result: HRAnalysis = structured_llm.invoke(prompt)
    is_hr = result["is_hr_question"]

    # HR 여부에 따라 answer_type 세팅
    answer_type = "pending" if is_hr else "reject"

    return cast(State, {**state, "is_hr_question": is_hr, "answer_type": answer_type})