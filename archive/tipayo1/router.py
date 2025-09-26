# router.py
import json
from typing import Dict, TypedDict, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from state import State
from llm import get_llm

# 부서 정보 dict (이메일과 슬랙 채널)
DEPARTMENTS = {
    "재무": {"name": "재무", "email": "fi@gaida.play.com", "slack": "#ask-fi"},
    "총무": {"name": "총무", "email": "ga@gaida.play.com", "slack": "#ask-ga"},
    "인프라": {"name": "인프라", "email": "in@gaida.play.com", "slack": "#ask-in"},
    "보안": {"name": "보안", "email": "se@gaida.play.com", "slack": "#ask-se"},
    "인사": {"name": "인사", "email": "hr@gaida.play.com", "slack": "#ask-hr"},
}

# =========================
# 1차 라우터: HR Router
# =========================
class HRAnalysis(TypedDict):
    is_hr_question: bool

def hr_router(state: State) -> State:
    """
    HR 여부만 판별, 그 결과를 상태에 저장
    """
    prompt = f"""
질문: "{state.get('refined_question') or state.get('user_question') or ''}"
당신은 "가이다 플레이 스튜디오(GPS)"의 HR 정책 안내 챗봇입니다.
아래 질문이 HR(인사/근무/휴가/복지/장비·보안/출장·비용처리 등) 관련인지 판별하세요.
JSON으로 {{ "is_hr_question": true|false }} 만 반환하세요.
""".strip()

    try:
        _llm = get_llm("router-lite")
        structured_llm = _llm.with_structured_output(HRAnalysis)
        result: HRAnalysis = structured_llm.invoke(prompt)
        is_hr = bool(result.get("is_hr_question", False))
    except Exception:
        # LLM 오류시 보수적으로 HR 처리
        is_hr = True

    answer_type = "pending" if is_hr else "reject"
    return cast(State, {**state, "is_hr_question": is_hr, "answer_type": answer_type})

def route_after_hr(state: State) -> str:
    """HR 판별 결과에 따라 다음 노드 결정"""
    return "router2" if state.get("is_hr_question") else "reject"

# =========================
# Answer_type: Reject
# =========================
def reject_node(state: State):
    return {"final_answer": "입력하신 질문은 HR 관련 문의가 아닙니다. HR 관련 질문만 가능합니다."}

# =========================
# 2차 라우터: RAG vs Department
# =========================
def _classify_rag_or_department(question: str) -> Dict[str, str]:
    """LLM을 사용한 통합 분류: RAG vs 담당자 안내 + 부서 결정"""
    llm = get_llm("router")

    system_prompt = """
당신은 사내(gaida play studio) HR 챗봇의 질문 분류 전문가입니다.
사용자의 질문을 분석하여 어떻게 처리할지 결정해주세요.

# 분류 기준
## 1. RAG 처리 대상 (route: "rag")
- 내부 규정/정책/제도 일반 질문, 문서에서 답 가능

## 2. 담당자 안내 대상 (route: "department")
- 개인별 처리/승인/신고/민감 상담

부서: 재무, 총무, 인프라, 보안, 인사
응답은 JSON만:
- RAG: {"route": "rag"}
- 담당자: {"route": "department", "department": "부서명"}
""".strip()

    user_prompt = f'사용자 질문: "{question}"\nJSON만 출력'
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    try:
        response = llm.invoke(messages)
        response_text = response.content.strip()
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            if "rag" in response_text.lower():
                return {"route": "rag"}
            for dept in DEPARTMENTS.keys():
                if dept in response_text:
                    return {"route": "department", "department": dept}
            return {"route": "department", "department": "인사"}
    except Exception:
        return {"route": "department", "department": "인사"}

def router2_node(state: State) -> State:
    """LLM 기반 질문 분류 및 라우팅 상태 업데이트"""
    question = state.get('refined_question') or state.get('user_question') or ""
    classification_result = _classify_rag_or_department(question)
    route = classification_result.get("route")
    department_name = classification_result.get("department")

    if route == "rag":
        return cast(State, {**state, "is_rag_suitable": True, "department_info": None, "answer_type": "rag_answer"})
    else:
        department_info = DEPARTMENTS.get(department_name, DEPARTMENTS["인사"])
        return cast(State, {**state, "is_rag_suitable": False, "department_info": department_info, "answer_type": "department_contact"})

def route_after_router2(state: State) -> Literal["rag", "department"]:
    """RAG 사용 여부에 따라 다음 노드 결정"""
    return "rag" if state.get('is_rag_suitable') else "department"
