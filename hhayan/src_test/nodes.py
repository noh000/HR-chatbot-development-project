# nodes.py

from dotenv import load_dotenv
load_dotenv()

from typing import List, Tuple
from langchain_core.documents import Document
from state import State
from llm import get_llm
from db import get_vectorstore  # 필요하면 get_retriever로 교체 가능

# messages/state에서 사용자 질문을 안전하게 추출
def _get_question(state: State) -> str:
    q = (state.get("user_question") or "").strip()
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

# 1) 검색 쿼리 정제
def analyze_query(state: State) -> dict:
    llm = get_llm("gen")
    question = _get_question(state)

    prompt = f"""
    당신은 "가이다 플레이 스튜디오(GPS)" HR 챗봇의 전처리 노드입니다.
    사용자의 질문을 정제해 주세요.
    규칙:
    1. 언어 규칙
     - 기본 언어는 한국어여야 합니다.
     - 한국어 문맥 안에 숫자나 일부 영어 단어(point, vacation 등)가 섞여 있는 경우는 허용합니다.
     - 한국어 없이 전부 영어로만 입력된 경우는 "invalid_input"으로 분류합니다.
    2. 형식 정리
     - 불필요한 특수문자는 제거합니다.
     - 문장의 의미를 전달하는 기본 문장부호(?, !, ., ,)는 보존합니다.
     - 여러 개의 공백은 하나의 공백으로 줄입니다.
    3. 표현 표준화
    문맥을 파악하여 HR 용어를 표준화합니다.
    표준화 예시:
        - "쉬려고 하는데 하루에 반만" → "반차 안내"
        - "컴퓨터 로그인이 안 돼" → "계정 보안 문제"
        - "회사 동호회 돈 지원해줘?" → "사내 동호회 지원"
        - "출근 좀 늦게 해도 돼?" → "시차 출근 제도"
        - "급여일이 언제야?" → "급여일 안내"
        - "복지 point 얼마지? 1000포인트인가?" → "복지 포인트 안내"
        - "나 반          차 쓸 수 있어?" → "반차 안내"
        동의어, 유의어, 줄임말, 초성 표현도 표준화 합니다.
        예시:
        - "대휴" → "대체휴가"
        - "ㄱㄱ" → "고고"
        - "ㅇㅇ" → "응응"
        - "내규" → "내부규칙"

    사용자 질문:
        {question}
    위 규칙으로 불필요한 내용은 제거하고 출력하라.
    """.strip()

    result = llm.invoke(prompt).content.strip() if question else ""
    return {"refined_question": result}

# 2) 문서 검색
def retrieve(state: State) -> dict:
    vs = get_vectorstore(index_name="gaida-company-rules", file_path="04_복지정책_v1.0.md")
    refined_question = state.get("refined_question", "") or _get_question(state) or ""
    if not refined_question:
        return {"retrieved_docs": []}
    docs = vs.similarity_search(refined_question, k=5)
    return {"retrieved_docs": docs}

# 3) 재순위화(정규식 기반 파싱 유지)
def rerank(state: State) -> dict:
    llm = get_llm("gen")
    question = _get_question(state)
    if not question or not state.get("retrieved_docs"):
        return {"retrieved_docs": state.get("retrieved_docs", [])}

    import re
    scored: List[Tuple[Document, float]] = []

    for doc in state.get("retrieved_docs", []):
        prompt = f"""
        질문: "{question}"
        문서 내용: "{doc.page_content}"
        0~1 사이 숫자로 관련도만 출력:
        """.strip()
        txt = (llm.invoke(prompt).content or "").strip()
        cleaned = txt.replace(",", ".")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
        try:
            score = float(m.group()) if m else 0.0
        except Exception:
            score = 0.0
        score = max(0.0, min(1.0, score))
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored[:3]]
    return {"retrieved_docs": top_docs}

# 4) 답변 생성
def generate_answer(state: State) -> dict:
    llm = get_llm("gen")
    question = _get_question(state)
    if not question:
        return {"final_answer": "문서에 근거가 없어 확답하기 어렵습니다. 질문을 제공해 주세요."}

    context = ""
    for i, doc in enumerate(state.get("retrieved_docs", []), start=1):
        context += f"[{i}] ({doc.metadata.get('source', 'unknown')})\n{doc.page_content}\n\n"

    if not context.strip():
        return {"final_answer": "문서에 근거가 없어 확답하기 어렵습니다. 관련 출처가 검색되지 않았습니다."}

    prompt = f"""
    아래 출처 문서만 근거로 질문에 답하세요. 문서에 명시가 없으면 "문서에 근거가 없어 확답하기 어렵습니다."라고 답하세요.
    본문 중 인용한 부분 뒤에 [출처 번호]를 붙이고, 답변 마지막에 출처 목록을 정리해주세요.
    질문: "{question}"
    {context}
    답변:
    """.strip()

    answer = llm.invoke(prompt).content.strip()
    return {"final_answer": answer}

# 5) 답변 검증
def verify_answer(state: State) -> dict:
    llm = get_llm("gen")
    sources = ", ".join(doc.metadata.get("source", "unknown") for doc in state.get("retrieved_docs", []))
    prompt = f"""
    아래 답변이 출처들[{sources}]의 내용과 일치하는지 '일치함' 또는 '불일치함'만 답하세요.
    답변: "{state.get("final_answer", "")}"
    판단:
    """.strip()
    verdict = llm.invoke(prompt).content.strip()
    verdict = "일치함" if verdict.startswith("일치") else ("불일치함" if verdict.startswith("불일치") else "불일치함")
    return {"verification": verdict}

# 6) 담당자 안내 답변 생성
def department_node(state: State) -> dict:
    """
    담당자 안내 응답 생성
    """
    department = state.get('department_info') 

    if not department:
        # 기본값: 인사팀
        department = {"name": "인사", "email": "hr@gaida.play.com", "slack": "#ask-hr"}
    
    response = f"""
해당 문의사항은 **{department['name']}팀**으로 문의하시면 정확하고 빠른 답변을 받으실 수 있습니다.

📧 **이메일**: {department['email']}
💬 **슬랙**: {department['slack']}

추가 질문이 있으시면 언제든 말씀해 주세요! 😊
    """.strip()
    
    return {"final_answer": response}
