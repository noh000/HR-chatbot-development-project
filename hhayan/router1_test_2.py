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
    당신은 HR과 관련 있는 질문을 분류하는 어시스턴트입니다.
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
