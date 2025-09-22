# router1 router

# =========================
# 3. HR Router
# =========================
def route_after_hr(state: State) -> str:
    """HR 판별 결과에 따라 다음 노드 결정"""
    return "router2" if state["next_step"] == "router2" else "reject"