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
    역할: 검색 쿼리 엔지니어
    다음 사용자의 복지 관련 질문을 가이다 플레이 스튜디오 복지정책 문서(04_복지정책_v1.0.md) 검색에 최적화된 한 줄 쿼리로 변환하라.
    한국어 공식 정책 용어만 사용하고, 숫자는 아라비아 숫자로 표준화하라.
    섹션은 선두에 두고, 핵심 키워드는 콤마(,)로 구분하라.
    동의어/관련어는 | 로 확장하되 2~4개로 제한하고, 불필요한 설명은 절대 출력하지 마라.
    수치(일수, 금액)와 조건(예: 출근율 80%, 진단서 제출, 사용 기한 등)이 질문에 있으면 포함하라.
    섹션 후보(문서 용어): 연차휴가, 병가, 가족돌봄휴가, 복지포인트, 교육비 지원, 장비 지원, 건강검진, 카페/스낵바, 동아리 활동, 임신·출산·육아
    용어 표준화 예: “언제까지 사용”→“연차 사용 기한”, 월차→월차휴가, 복지포인트→포인트, 교육비→교육비 지원
    출력 형식(한 줄, 라벨/따옴표/불릿 금지, 마침표 금지):
    섹션, 키워드1|동의어1|동의어2, 키워드2, 수치/조건...
    예시(출력 예, 실제 출력에 포함하지 말 것):
    연차휴가, 연차 사용 기한|미사용 연차 소멸, 발생연차 기준, 출근율 80%
    복지포인트, 포인트 금액|포인트, 사용 기한, 연간 한도 50만
    사용자 질문:
    {question}
    위 형식으로 최종 한 줄 쿼리만 출력하라.
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
