# router1 state

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
