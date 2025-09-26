# nodes.py

from dotenv import load_dotenv
load_dotenv()

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from state import State
from llm import get_llm
from db import get_vectorstore

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

# 1) 사용자 질문 정제
def refine_question(state: State) -> dict:
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
    return {
        "user_question": question,
        "refined_question": result
    }


# 2) 문서 검색
def retrieve(state: State) -> dict:
    # 미리 생성된 Pinecone 인덱스에 연결하여 retriever를 생성합니다.
    # db.py의 get_vectorstore 함수를 사용하여 기존 인덱스를 가져옵니다.
    vs = get_vectorstore(index_name="gaida-hr-rules")
    
    refined_question = state.get("refined_question", "") or _get_question(state) or ""
    if not refined_question:
        # 질문이 없으면 빈 리스트를 반환합니다.
        return {"retrieved_docs": []}
        
    # retriever를 사용하여 유사도 높은 문서를 3개 검색합니다.
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(refined_question)
    
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
def generate_rag_answer(state: State) -> dict:
    llm = get_llm("gen")
    question = _get_question(state)
    if not question:
        return {"final_answer": "문서에 근거가 없어 답변드리기 어렵습니다. 다시 질문해주세요."}

    context = ""
    for i, doc in enumerate(state.get("retrieved_docs", []), start=1):
        context += f"[{i}] ({doc.metadata.get('source', 'unknown')})\n{doc.page_content}\n\n"

    if not context.strip():
        return {"final_answer": "문서에 근거가 없어 답변드리기 어렵습니다. 관련 출처가 검색되지 않았습니다."}

    prompt = f"""
    당신은 "가이다 플레이 스튜디오(GPS)"의 친절한 HR 정책 안내 챗봇입니다.
    아래 출처 문서 내용만을 근거로 해서 질문에 대해 명확하고 간결하게 답변하세요.
    문서에 명시된 내용이 없으면 "문서에 근거가 없어 답변드리기 어렵습니다."라고 답해야 합니다.
    답변 본문 중 인용한 부분이 있다면, 문장 끝에 [출처 번호]를 붙여주세요.
    답변의 마지막에는 '출처 목록'을 정리해서 보여주세요.

    # 질문
    {question}

    # 출처 문서
    {context}

    # 답변
    """.strip()

    answer = llm.invoke(prompt).content.strip()
    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer
    }

# 5) 답변 검증
def verify_rag_answer(state: State) -> dict:
    llm = get_llm("gen")

    # [수정] 검증을 위해 문서의 '이름'이 아닌 '내용'을 컨텍스트로 구성합니다.
    context = ""
    for doc in state.get("retrieved_docs", []):
        context += f"- {doc.page_content}\n"

    final_answer = state.get("final_answer", "")

    # [추가] 컨텍스트나 답변이 없으면 검증이 무의미하므로 '불일치함'으로 처리합니다.
    if not context.strip() or not final_answer.strip():
        return {"verification": "불일치함"}

    prompt = f"""
    당신은 생성된 답변이 주어진 문서 내용에만 근거했는지 검증하는 AI 평가자입니다.
    '답변'이 아래 '문서' 내용과 완전히 일치하는 경우에만 '일치함'을, 조금이라도 다르거나 관련 없는 내용이 있다면 '불일치함'을 출력하세요.
    다른 어떤 설명도 추가하지 말고, '일치함' 또는 '불일치함' 두 단어 중 하나로만 답변해야 합니다.

    # 문서
    {context}

    # 답변
    "{final_answer}"

    # 판단 (일치함/불일치함):
    """.strip()

    verdict = llm.invoke(prompt).content.strip()

    # [개선] LLM이 지시를 어기고 "네, 일치합니다."와 같이 답변해도 처리 가능
    if "일치함" in verdict:
        final_verdict = "일치함"
    else:
        final_verdict = "불일치함"
    return {"verification": final_verdict}

# 6) 담당자 안내 답변 생성
def generate_contact_answer(state: State) -> dict:
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
    
    return {
        "messages": [AIMessage(content=response)],
        "final_answer": response
    }