# router1 state

# =========================
# 1. State
# =========================
class State(MessagesState, total=False):
    good_question: str                      # 전처리 후 질문 내용 
    is_hr_question: Literal["yes", "no"]   # HR 여부
    next_step: Literal["router2", "reject"] # 다음 노드 방향
    answer: str                        # 최종 답변 (추가됨)