# LLM 인스턴스와 각 노드(함수)를 정의합니다.

# node.py
import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import TextLoader

from state.py import State  # 같은 디렉토리에 위치한다고 가정

# LLM 모델 설정
llm = ChatOpenAI(model='gpt-4.1', temperature=0)

def analyze_query(state: State) -> dict:
    prompt = (
        "사용자의 복지 관련 질문을 가이다 플레이 스튜디오 복지정책 문서(04_복지정책_v1.0.md)에서\n"
        "검색 가능한 최적화된 쿼리로 변환하세요.\n\n"
        "변환 규칙:\n"
        "1. 문서 섹션 명시 (연차휴가, 병가, 가족돌봄휴가, 복지포인트, 교육비 지원, 장비 지원, 건강검진, 카페/스낵바, 동아리 활동, 임신·출산·육아 등)\n"
        "2. 핵심 키워드(예: 연차일수, 출근율, 포인트 금액, 지원 한도 등) 추출 및 표준화\n"
        "3. 수치(일수, 금액) 및 조건(출근율 80%, 진단서 제출, 월차 사용 기간 등) 정보 포함\n"
        "4. 구어체 표현을 공식 정책 용어로 변환 (예: “언제까지 사용해야 하나요?” → “연차 사용 기한”)\n"
        "5. 동의어 및 관련 용어 확장 (월차→월차휴가, 복지포인트→포인트, 교육비→교육비 지원 등)\n\n"
        f"\"\"\"\n{state.question}\n\"\"\"\n\n"
        "변환된 쿼리만 출력해주세요."
    )
    result = llm(prompt).strip()
    return {"query": result}

def setup_vectorstore(state: State, file_path: str = "04_복지정책_v1.0.md", index_name: str = 'gaida-company-rules') -> dict:
    if not os.path.exists(file_path):
        logging.error(f"{file_path} 파일이 없습니다.")
        raise FileNotFoundError(f"{file_path} 파일이 없습니다.")
    loader = TextLoader(file_path, encoding="utf-8")
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(raw_docs)
    for idx, doc in enumerate(split_docs):
        setattr(doc, 'source', f"{file_path}#chunk-{idx}")
    embedding = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = PineconeVectorStore.from_documents(
        split_docs,
        index_name=index_name,
        embedding=embedding
    )
    return {"vectorstore": vectorstore}

def retrieve(state: State) -> dict:
    vs = state.vectorstore
    docs = vs.similarity_search(state.query, k=5)
    return {"docs": docs}

def rerank(state: State) -> dict:
    scored = []
    for doc in state.docs:
        prompt = (
            f"질문: \"{state.question}\"\n\n"
            f"문서 내용: \"{doc.page_content}\"\n\n"
            "0~1 사이 숫자로 관련도만 출력:"
        )
        txt = llm(prompt).strip()
        try:
            score = float(txt)
        except ValueError:
            score = 0.0
        scored.append((doc, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored[:3]]
    return {"docs": top_docs}

def generate_answer(state: State) -> dict:
    context = ""
    for i, doc in enumerate(state.docs, start=1):
        context += f"[{i}] ({getattr(doc, 'source')})\n{doc.page_content}\n\n"
    prompt = (
        f"아래 출처 문서를 참고해 질문에 답하세요. 본문 중 인용 뒤에 [출처 번호]를 붙이고, "
        "답변 마지막에 출처 목록을 정리해주세요.\n\n"
        f"질문: \"{state.question}\"\n\n"
        f"{context}"
        "답변:"
    )
    answer = llm(prompt).strip()
    return {"result": answer}

def verify_answer(state: State) -> dict:
    sources = ", ".join(getattr(doc, 'source') for doc in state.docs)
    prompt = (
        f"아래 답변이 출처들[{sources}]와 일치하는지 '일치함' 또는 '불일치함'만 답하세요.\n\n"
        f"답변: \"{state.result}\"\n\n"
        "판단:"
    )
    verdict = llm(prompt).strip()
    return {"verification": verdict}
