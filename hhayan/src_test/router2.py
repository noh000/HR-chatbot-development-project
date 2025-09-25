from typing import Dict, List, Optional, TypedDict, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from state import State


# 부서 정보 dict (이메일과 슬랙 채널)
DEPARTMENTS = {
    "재무": {"name": "재무", "email": "fi@gaida.play.com", "slack": "#ask-fi"},
    "총무": {"name": "총무", "email": "ga@gaida.play.com", "slack": "#ask-ga"},
    "인프라": {"name": "인프라", "email": "in@gaida.play.com", "slack": "#ask-in"},
    "보안": {"name": "보안", "email": "se@gaida.play.com", "slack": "#ask-se"},
    "인사": {"name": "인사", "email": "hr@gaida.play.com", "slack": "#ask-hr"},
}

class SecondaryRouter:
    """2차 라우터: LLM 기반 RAG vs 담당자 안내 분류 시스템"""
    
    def __init__(self, llm_model: ChatOpenAI):
        self.llm = llm_model

    def classify_with_llm(self, question: str) -> Dict[str, str]:
        """
        LLM을 사용한 통합 분류: RAG vs 담당자 안내 + 부서 결정
        """
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
        - **인사**: 개별 급여 문의, 채용, 인사평가, 퇴직, 퇴직금, 입사, 퇴사, 평가, 승진, 개인적 근무 상담

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
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # JSON 파싱 시도
            import json
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

    def route_question(self, state: State) -> State:
        """
        LLM 기반 질문 분류 및 라우팅
        """
        question = state['refined_question']
        
        print(f" LLM 기반 질문 분류 시작...")
        
        # LLM을 통한 통합 분류
        classification_result = self.classify_with_llm(question)
        
        route = classification_result.get("route")
        department = classification_result.get("department")
        
        print(f" 분류 결과: {classification_result}")
        
        if route == "rag":
            # RAG 처리로 분류
            state['is_rag_suitable'] = True 
            state['department_info'] = None
            state['answer_type'] = "rag_answer"  # RAG 처리 상태
            print("➡️ RAG 시스템으로 라우팅")
        else:
            # 담당자 안내로 분류
            state['is_rag_suitable'] = False 
            state['department_info'] = DEPARTMENTS.get(department, DEPARTMENTS["인사"])
            state['answer_type'] = "department_contact"
            print(f"➡️ {department}팀 담당자 안내로 라우팅")
        
        return state

    def generate_department_response(self, state: State) -> str:
        """
        담당자 안내 응답 생성
        """
        department = state.get('department_info') 

        if not department:
            return "해당 문의사항은 인사팀으로 문의하시면 정확하고 빠른 답변을 받으실 수 있습니다.\n\n(hr@gaida.play.com / #ask-hr)\n\n추가 질문이 있으시면 언제든 말씀해 주세요! 😊"
        
        response = f"""
        해당 문의사항은 **{department['name']}팀**으로 문의하시면 정확하고 빠른 답변을 받으실 수 있습니다.

        📧 **이메일**: {department['email']}
        💬 **슬랙**: {department['slack']}
        
        추가 질문이 있으시면 언제든 말씀해 주세요! 😊
        """
        
        return response.strip()

    def should_use_rag(self, state: State) -> bool:
        """
        RAG 사용 여부 판단
        """
        return state.get('is_rag_suitable', False) 

    def process_secondary_routing(self, state: State) -> State:
        """
        2차 라우터 전체 처리 프로세스
        """
        # 1. 질문 분류 및 라우팅
        state = self.route_question(state)
        
        # 2. 담당자 안내인 경우 최종 답변 생성
        if state['answer_type'] == "department_contact":
            state['final_answer'] = self.generate_department_response(state)
            
        return state
