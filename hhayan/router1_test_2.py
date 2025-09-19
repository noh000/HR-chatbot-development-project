from dotenv import load_dotenv
from typing_extensions import Literal
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

class Router1State(TypedDict):
    question: str
    is_hr_question: Literal["yes", "no"]
    next_step: Literal["rag", "reject"]

def router1_node(state: Router1State) -> Router1State:
    prompt = f"""
    당신은 "가이다 플레이 스튜디오(GPS)"의 HR 정책 안내 챗봇입니다.  
    당신은 회사 내부 직원의 질문이 **HR(인사/근무/휴가/복지/장비·보안/출장·비용처리 등)**과 관련된 질문인지 아닌지를 판별하는 것입니다.  

    ## 참고
    - 근무: 하이브리드(재택) 등  
    - 휴가 관련  
    - 장비·보안: 노트북/모니터/장비 보조금, 계정 보안, 도난 절차 등  
    - 복지: 복지포인트, 교육비, 사내 동아리/동호회/모임 지원, 상담 지원, 건강검진 지원,
      카페/스낵바(사무실 내 무료 간식, 커피머신, 음료, 과자, 컵라면 등), 임신/출산/육아 지원 등  
    - 기타: 출장비, 식대, 일비, 급여일, 교통비 등  

    ## 금지
    - 개인 HR 데이터(개인 연차 잔여일, 급여·퇴직금 계산 등)  
    
    출력 형식은 반드시 아래 두 값 중 하나로만 하세요:
    출력 형식:
    is_hr_question: yes | no
    next_step: rag | reject
    질문: "{state['question']}"
    """
    response = llm.invoke(prompt).content.strip()
    
    is_hr, next_step = "no", "reject"
    
    for line in response.splitlines():
        if line.startswith("is_hr_question:"):
            is_hr = line.split(":")[1].strip()
        if line.startswith("next_step:"):
            next_step = line.split(":")[1].strip()
    return {**state, "is_hr_question": is_hr, "next_step": next_step}

def route_after_router1(state: Router1State) -> str:
    return END if state["next_step"] == "rag" else "reject"

def reject_node(state: Router1State) -> dict:
    return {**state, "answer": "❌ 지원하지 않는 질문입니다. HR 관련 문의만 가능합니다."}

builder = StateGraph(Router1State)
builder.add_node("router1", router1_node)
builder.add_node("reject", reject_node)
builder.add_conditional_edges("router1", route_after_router1, {END: END, "reject": "reject"})
builder.add_edge("reject", END)
builder.add_edge(START, "router1")

graph = builder.compile()
