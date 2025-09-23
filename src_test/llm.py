import os
from langchain_openai import ChatOpenAI

# 공용 LLM 팩토리: 역할별 모델 지정
def get_llm(role: str = "gen") -> ChatOpenAI:
    """
    role:
    - "gen": 본문 생성/분석 (기본 gpt-4.1)
    - "router": 2차 라우터 (기본 gpt-4.1-mini)
    - "router-lite": 1차 HR 분류 (기본 gpt-4.1-nano)
    환경변수로 덮어쓰기:
    GEN_LLM, ROUTER_LLM, ROUTER_LITE_LLM
    """
    model_map = {
        "gen": os.getenv("GEN_LLM", "gpt-4.1"),
        "router": os.getenv("ROUTER_LLM", "gpt-4.1-mini"),
        "router-lite": os.getenv("ROUTER_LITE_LLM", "gpt-4.1-nano"),
    }
    model = model_map.get(role, model_map["gen"])
    return ChatOpenAI(model=model, temperature=0)