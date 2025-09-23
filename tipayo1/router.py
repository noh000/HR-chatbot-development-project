# router.py

from typing import Dict
from typing_extensions import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import State

# 1) 구조화 출력 스키마(1차 HR 라우터)
class HRClassification(BaseModel):
    is_hr_question: Literal["yes", "no"]
    next_step: Literal["router2", "reject"]

# 2) 경량 분류 LLM (비용/속도 최적화)
_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# 메시지에서 질문 추출
def _extract_question(state: State) -> str:
    q = (state.get("question") or "").strip()
    if q:
        return q

    msgs = state.get("messages") or []
    for m in reversed(list(msgs)):
        content = None
        role = None

        if hasattr(m, "content"):
            content = getattr(m, "content", None)
            role = getattr(m, "type", None) or getattr(m, "role", None)

        if isinstance(m, dict):
            content = m.get("content", content)
            role = m.get("type") or m.get("role") or role

        if isinstance(content, str) and content.strip():
            if role in (None, "human", "user"):
                return content.strip()

    return ""

# 3) HR 분류 노드
def hr_node(state: State) -> State:
    structured = _llm.with_structured_output(HRClassification)

    category = (state.get("category") or "미분류").strip()
    question = _extract_question(state)

    prompt = f"""
    당신은 "가이다 플레이 스튜디오(GPS)"의 HR 안내 챗봇입니다.
    회사 내부 직원의 질문이 HR과 관련된 질문인지 아닌지를 판별하세요.

    ### 참고 정보
    - 카테고리: "{category}"
    - 질문: "{question}"

    ### 출력 형식 (반드시 JSON):
    {{
      "is_hr_question": "yes" | "no",
      "next_step": "router2" | "reject"
    }}
    """.strip()

    try:
        result: HRClassification = structured.invoke(prompt)
        is_hr = result.is_hr_question
        next_step = result.next_step
    except Exception:
        is_hr = "no"
        next_step = "reject"

    return {
        **state,
        "is_hr_question": is_hr,
        "next_step": next_step,
    }

# 4) 조건 분기 함수
def route_after_hr(state: State) -> str:
    return state.get("next_step") or "reject"

# 5) 거절 터미널 노드
def reject_node(state: State) -> State:
    msg = "HR 문의로 분류되지 않아 일반 워크플로로 보내지 않습니다. 질문을 구체화하거나 HR 범주로 다시 시도해 주세요."
    return {**state, "result": msg, "verification": "불일치함"}

# ---------------------------
# 2차 라우터: RAG vs 담당자 안내
# ---------------------------
DEPARTMENTS: Dict[str, Dict[str, str]] = {
    "재무": {"name": "재무", "email": "fi@gaida.play.com", "slack": "#ask-fi"},
    "총무": {"name": "총무", "email": "ga@gaida.play.com", "slack": "#ask-ga"},
    "인프라": {"name": "인프라", "email": "in@gaida.play.com", "slack": "#ask-in"},
    "보안": {"name": "보안", "email": "se@gaida.play.com", "slack": "#ask-se"},
    "인사": {"name": "인사", "email": "hr@gaida.play.com", "slack": "#ask-hr"},
}

_llm_router2 = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

def _classify_rag_or_department(question: str) -> Dict[str, str]:
    system_prompt = """
    당신은 사내(gaida play studio) HR 챗봇의 질문 분류 전문가입니다.
    사용자의 질문을 분석하여 어떻게 처리할지 결정해주세요.

    # 분류 기준
    ## 1. RAG 처리 대상 (route: "rag")
    - 회사 내부 규정, 정책, 제도에 대한 일반적인 질문
    - 문서에서 답변을 찾을 수 있는 정보성 질문

    ## 2. 담당자 안내 대상 (route: "department")
    - 개인별 맞춤 처리가 필요한 질문
    - 실시간 처리나 승인이 필요한 업무
    - 문제 해결이나 신고가 필요한 상황
    - 개별 상담이 필요한 민감한 사안

    ### 부서별 담당 업무:
    - 재무: 급여, 세금, 예산, 회계, 지출, 송금, 계산서, 청구서, 지급, 비용, 환급
    - 총무: 사무실, 비품, 물품, 구매, 수령, 우편, 사무용품, 시설, 행사, 차량, 청소, 자산, 출장, 숙박, 교통
    - 인프라: 서버, 네트워크, 컴퓨터, IT, 소프트웨어, 장비, 시스템, 접속, VPN, 계정, 접근
    - 보안: 보안, 해킹, 정보, 유출, 침해, 랜섬웨어, 백신, 데이터, 비밀번호, 방화벽, 악성코드, 암호
    - 인사: 개별 급여 문의, 채용, 인사평가, 퇴직, 입사, 퇴사, 평가, 승진, 개인적 근무 상담

    # 응답 형식
    RAG 처리인 경우:
    {"route": "rag"}

    담당자 안내인 경우:
    {"route": "department", "department": "부서명"}

    부서명은 반드시 다음 중 하나여야 합니다: 재무, 총무, 인프라, 보안, 인사
    부득이하게 재무, 총무, 인프라, 보안 부서에 해당하지 않을 경우에는 인사로 지정해주세요.
    """.strip()

    user_prompt = f'사용자 질문: "{question}"\n위 질문을 분석하여 RAG 처리할지, 담당자 안내할지 결정하고 JSON 형식으로 응답해주세요.'
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    try:
        response = _llm_router2.invoke(messages)
        response_text = (response.content or "").strip()

        import json
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            if "rag" in response_text:
                return {"route": "rag"}
            for dept in DEPARTMENTS.keys():
                if dept in response_text:
                    return {"route": "department", "department": dept}
            return {"route": "department", "department": "인사"}
    except Exception:
        return {"route": "department", "department": "인사"}

def router2_node(state: State) -> State:
    """2차 라우터: RAG vs 담당자 안내"""
    question = _extract_question(state)
    result = _classify_rag_or_department(question)
    route = result.get("route", "department")
    dept_name = result.get("department")

    if route == "rag":
        return {**state, "is_rag": True, "department": None}
    else:
        dept = DEPARTMENTS.get(dept_name or "", DEPARTMENTS["인사"])
        return {**state, "is_rag": False, "department": dept}

def route_after_router2(state: State) -> str:
    """rag 또는 department로 분기 키 반환"""
    return "rag" if state.get("is_rag") else "department"

def department_node(state: State) -> State:
    """담당자 안내 메시지 생성 (터미널)"""
    dept = state.get("department")
    if not dept:
        text = (
            "해당 문의사항은 인사팀으로 문의하시면 정확하고 빠른 답변을 받으실 수 있습니다.\n\n"
            "(hr@gaida.play.com / #ask-hr)\n\n"
            "추가 질문이 있으시면 언제든 말씀해 주세요! 😊"
        )
        return {**state, "result": text}

    text = f"""
    해당 문의사항은 **{dept['name']}팀**으로 문의하시면 정확하고 빠른 답변을 받으실 수 있습니다.
    📧 이메일: {dept['email']}
    💬 슬랙: {dept['slack']}
    추가 질문이 있으시면 언제든 말씀해 주세요! 😊
    """.strip()
    return {**state, "result": text}
