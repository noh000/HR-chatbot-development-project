# router1 state

# =========================
# 1. State 정의
# =========================
class State(MessagesState, total=False):
    refined_question: str                   # LLM이 정제한 질문 (문맥 보완, 맞춤법 교정 등)
    is_hr_question: bool                    # HR 여부 (true, false)
    next_step: Literal["router2", "reject"] # 다음 노드 방향