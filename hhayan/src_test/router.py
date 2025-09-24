# router.py

import json
from typing import Dict, List, Optional, TypedDict, Literal, cast
from langchain_openai import ChatOpenAI
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

    정제 질문: "{state['refined_question']}"

    당신은 "가이다 플레이 스튜디오(GPS)"의 HR 정책 안내 챗봇입니다.
    정제 질문이 HR(인사/근무/휴가/복지/장비·보안/출장·비용처리 등) 관련인지 판별하세요.

    금지 규칙: 금지 규칙에 해당되면 반드시 is_hr_question을 false로 판별하세요
     - 개인 HR 데이터 (예: "내 급여", "내 퇴직금")
     - 개인정보 (예: 주민등록번호, 사원번호, 이름)
     - 회사 내부 보안 내용 (예: 회사 재정 상황, 신규 프로젝트, 회사의 중요한 내부 문건)
     - 복잡한 급여·퇴직금 계산 요청
     - 법률 자문 요청이나 법률 상담 톤의 질문
    """
    _llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    structured_llm = _llm.with_structured_output(HRAnalysis)

    result: HRAnalysis = structured_llm.invoke(prompt)
    is_hr = result["is_hr_question"]

    # HR 여부에 따라 answer_type 세팅
    answer_type = "pending" if is_hr else "reject"

    return cast(State, {**state, "is_hr_question": is_hr, "answer_type": answer_type})

# 라우터 함수. Return 은 다음으로 이어질 Node 의 이름
def route_after_hr(state: State) -> str:
    """HR 판별 결과에 따라 다음 노드 결정"""
    # HR 질문이면 router2로, 아니면 reject로
    return "router2" if state["is_hr_question"] else "reject"

# =========================
# Answer_type: Reject
# =========================
def reject_node(state: State):
    """HR 관련이 아닌 질문에 대한 거부 메시지"""
    # return "입력하신 질문은 HR 관련 문의가 아닙니다. HR 관련 질문만 가능합니다."
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
    - 회사 내부 규정, 정책, 제도에 대한 일반적인 질문
    - 문서에서 답변을 찾을 수 있는 정보성 질문
    - 예시:
    * "연차 규정이 어떻게 되나요?"
    * "재택근무 정책을 알려주세요"
    * "복지제도에는 무엇이 있나요?"
    * "근무시간은 어떻게 되나요?"
    * "휴가 신청 방법을 알려주세요"
    * "장비 사용 규칙이 궁금해요"

    ## 2. 담당자 안내 대상 (route: "department")
    - 개인별 맞춤 처리가 필요한 질문
    - 실시간 처리나 승인이 필요한 업무
    - 문제 해결이나 신고가 필요한 상황
    - 개별 상담이 필요한 민감한 사안

    ### 부서별 담당 업무:
    - **재무**: 급여, 세금, 예산, 회계, 지출, 송금, 계산서, 청구서, 지급, 비용, 환급
    - **총무**: 사무실, 비품, 물품, 구매, 수령, 우편, 사무용품, 시설, 행사, 차량, 청소, 자산, 출장, 숙박, 교통
    - **인프라**: 서버, 네트워크, 컴퓨터, IT, 소프트웨어, 장비, 시스템, 접속, VPN, 계정, 접근
    - **보안**: 보안, 해킹, 정보, 유출, 침해, 랜섬웨어, 백신, 데이터, 비밀번호, 방화벽, 악성코드, 암호
    - **인사**: 개별 급여 문의, 채용, 인사평가, 퇴직, 입사, 퇴사, 평가, 승진, 개인적 근무 상담

    # 응답 형식
    다음 JSON 형식으로만 응답해주세요:

    RAG 처리인 경우:
    {"route": "rag"}

    담당자 안내인 경우:
    {"route": "department", "department": "부서명"}

    부서명은 반드시 다음 중 하나여야 합니다: 재무, 총무, 인프라, 보안, 인사

    부득이하게 재무, 총무, 인프라, 보안 부서에 해당하지 않을 경우에는 인사로 지정해주세요.
            """
            
    user_prompt = f"""
    사용자 질문: "{question}"

    위 질문을 분석하여 RAG 처리할지, 담당자 안내할지 결정하고 JSON 형식으로 응답해주세요.
            """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # JSON 파싱 시도
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # JSON 파싱 실패시 텍스트에서 추출
            if "rag" in response_text:
                return {"route": "rag"}
            else:
                # 부서명 추출 시도
                for dept in DEPARTMENTS.keys():
                    if dept in response_text:
                        return {"route": "department", "department": dept}
                return {"route": "department", "department": "인사"}
            
    except Exception as e:
        print(f"LLM 분류 오류: {e}")
        # 기본값: 인사팀 담당자 안내로 라우팅
        return {"route": "department", "department": "인사"}

def router2_node(state: State) -> State:
    """LLM 기반 질문 분류 및 라우팅 상태 업데이트"""
    question = state['refined_question']
    
    print(f" LLM 기반 질문 분류 시작...")
    
    # LLM을 통한 통합 분류
    classification_result = _classify_rag_or_department(question)
    
    route = classification_result.get("route")
    department_name = classification_result.get("department")
    
    print(f" 분류 결과: {classification_result}")
    
    if route == "rag":
        # RAG 처리로 분류
        print("➡️ RAG 시스템으로 라우팅")
        return cast(State, {**state, "is_rag_suitable": True, "department_info": None, "answer_type": "rag_answer"})
    else:
        # 담당자 안내로 분류
        department_info = DEPARTMENTS.get(department_name, DEPARTMENTS["인사"])
        print(f"➡️ {department_name}팀 담당자 안내로 라우팅")
        return cast(State, {**state, "is_rag_suitable": False, "department_info": department_info, "answer_type": "department_contact"})

def route_after_router2(state: State) -> Literal["rag", "department"]:
    """RAG 사용 여부에 따라 다음 노드 결정"""
    if state.get('is_rag_suitable'):
        return "rag"
    else:
        return "department"
