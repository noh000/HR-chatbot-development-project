# node.py

from dotenv import load_dotenv
load_dotenv()

from typing import List, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from state import State
from db import get_vectorstore

# 추가: 메시지에서 사용자 질문을 안전하게 추출
def _get_question(state: State) -> str:
    q = (state.get("question") or "").strip()
    if q:
        return q
    msgs = state.get("messages") or []
    # 뒤에서 앞으로 순회하여 마지막 사용자 발화 검색
    for m in reversed(list(msgs)):
        content = None
        role = None
        # BaseMessage 객체
        if hasattr(m, "content"):
            content = getattr(m, "content", None)
            role = getattr(m, "type", None) or getattr(m, "role", None)
        # dict 형태 메시지
        if isinstance(m, dict):
            content = m.get("content", content)
            role = m.get("type") or m.get("role") or role
        if isinstance(content, str) and content.strip():
            # role 정보가 human/user이거나 비어 있어도 마지막 발화를 질문으로 사용
            if role in (None, "human", "user"):
                return content.strip()
    return ""

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4.1", temperature=0)

# 쿼리정제 노드
def analyze_query(state: State) -> dict:
    llm = get_llm()
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
    return {"query": result}

# 리트리브 노드
def retrieve(state: State) -> dict:
    vs = get_vectorstore(
        index_name="gaida-company-rules",
        file_path="04_복지정책_v1.0.md",
    )
    query = state.get("query", "") or _get_question(state) or ""
    if not query:
        return {"docs": []}
    docs = vs.similarity_search(query, k=5)
    return {"docs": docs}

# 리랭크 노드
def rerank(state: State) -> dict:
    llm = get_llm()
    question = _get_question(state)
    if not question or not state.get("docs"):
        return {"docs": state.get("docs", [])}
    scored: List[Tuple[Document, float]] = []
    for doc in state.get("docs", []):
        prompt = f"""
질문: "{question}"
문서 내용: "{doc.page_content}"
0~1 사이 숫자로 관련도만 출력:
""".strip()
        txt = llm.invoke(prompt).content.strip()
        try:
            score = float(txt)
        except Exception:
            score = 0.0
        score = max(0.0, min(1.0, score))
        scored.append((doc, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored[:3]]
    return {"docs": top_docs}

# 답변생성 노드
def generate_answer(state: State) -> dict:
    llm = get_llm()
    question = _get_question(state)
    # 질문이 끝내 비어 있으면 사용자 피드백 제공
    if not question:
        return {"result": "문서에 근거가 없어 확답하기 어렵습니다. 질문을 제공해 주세요."}
    context = ""
    for i, doc in enumerate(state.get("docs", []), start=1):
        context += f"[{i}] ({doc.metadata.get('source', 'unknown')})\n{doc.page_content}\n\n"
    # 문서가 없으면 명시적 답변
    if not context.strip():
        return {"result": "문서에 근거가 없어 확답하기 어렵습니다. 관련 출처가 검색되지 않았습니다."}
    prompt = f"""
아래 출처 문서만 근거로 질문에 답하세요. 문서에 명시가 없으면 "문서에 근거가 없어 확답하기 어렵습니다."라고 답하세요.
본문 중 인용한 부분 뒤에 [출처 번호]를 붙이고, 답변 마지막에 출처 목록을 정리해주세요.
질문: "{question}"
{context}
답변:
""".strip()
    answer = llm.invoke(prompt).content.strip()
    return {"result": answer}

# 답변검증 노드
def verify_answer(state: State) -> dict:
    llm = get_llm()
    sources = ", ".join(doc.metadata.get("source", "unknown") for doc in state.get("docs", []))
    prompt = f"""
아래 답변이 출처들[{sources}]의 내용과 일치하는지 '일치함' 또는 '불일치함'만 답하세요.
답변: "{state.get("result", "")}"
판단:
""".strip()
    verdict = llm.invoke(prompt).content.strip()
    verdict = "일치함" if verdict.startswith("일치") else ("불일치함" if verdict.startswith("불일치") else "불일치함")
    return {"verification": verdict}
