# router1 router

# =========================
# 4. HR Router
# =========================
def route_after_hr(state: State) -> str:
    """HR 판별 결과에 따라 다음 노드 결정"""
    # HR 질문이면 router2로, 아니면 reject로
    return "router2" if state["is_hr_question"] else "reject"
