import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def get_llm(role: str = "gen") -> ChatOpenAI:
    """
    노드별로 적합한 LLM 모델을 반환하는 팩토리 함수
    
    Args:
        role (str): 역할별 모델 선택
            - "gen": 본문 생성/분석
            - "router1": 1차 라우터
            - "router2": 2차 라우터 
    
    Returns:
        ChatOpenAI: 설정된 LLM 인스턴스
        
    Environment Variables:
        - OPENAI_API_KEY: OpenAI API 키 (필수)
    """
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 환경변수에 설정되지 않았습니다.")
    
    # 역할별 고정 모델 매핑 (실제 존재하는 OpenAI 모델)
    model_map = {
        "gen": os.getenv("GEN_LLM", "gpt-4.1"),
        "router1": os.getenv("ROUTER1_LLM", "gpt-4.1-mini"),
        "router2": os.getenv("ROUTER2_LLM", "gpt-4.1-nano"),
    }
    
    # 역할에 맞는 모델 선택 (기본값: gen)
    model_name = model_map.get(role, model_map["gen"])
    
    try:
        return ChatOpenAI(
            model=model_name,
            temperature=0,  # 일관된 응답을 위해 0으로 설정
            api_key=api_key
        )
    except Exception as e:
        raise RuntimeError(f"LLM 초기화 실패 (role: {role}, model: {model_name}): {str(e)}")