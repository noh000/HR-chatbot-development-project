# router1 node: HR NODE + REJECT NODE

# =========================
# 2. HR NODE
# =========================
def hr_node(state: State) -> State:
    """HR 관련 질문인지 판별하는 노드"""
    prompt = f"""
    당신은 "가이다 플레이 스튜디오(GPS)"의 HR 정책 안내 챗봇입니다.  
    당신은 회사 내부 직원의 질문이 HR(인사/근무/휴가/복지/장비·보안/출장·비용처리 등)과 관련된 질문인지 아닌지를 판별하는 것입니다.  
    
    ### 출력 형식 (반드시 JSON):
    {{
      "is_hr_question": "yes" | "no",
      "next_step": "router2" | "reject"
    }}

    질문: "{state['question']}"
    """

    # LLM 응답(json형태)의 안전성을 보장
    response = llm.invoke(prompt).content.strip()

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        # LLM이 JSON을 못 주면 기본값으로 fallback
        parsed = {"is_hr_question": "no", "next_step": "reject"}

    return {
        **state,
        "is_hr_question": parsed.get("is_hr_question", "no"),
        "next_step": parsed.get("next_step", "reject"),
    }

# =========================
# 4. Reject Node
# =========================
def reject_node(state: State) -> State:
    """HR 관련이 아닌 질문에 대한 거부 메시지"""
    return {
        **state,
        "answer": "지원하지 않는 질문입니다. HR 관련 문의만 가능합니다."
    }