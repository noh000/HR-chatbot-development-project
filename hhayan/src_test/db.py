# db.py

from dotenv import load_dotenv
import os
import logging
import threading
import time
from typing import Tuple, List

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

load_dotenv()
logging.basicConfig(level=logging.INFO)

_VSTORE_CACHE = {}
_VSTORE_LOCK = threading.Lock()

def _pc() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        # 즉시 실패: API 키 누락 시 명확한 예외 발생
        raise RuntimeError("PINECONE_API_KEY 미설정: Pinecone 초기화를 진행할 수 없습니다.")
    return Pinecone(api_key=api_key)

def _index_exists(pc: Pinecone, name: str) -> bool:
    try:
        idx = pc.list_indexes()
        names = set()
        if hasattr(idx, "names"):
            names = set(idx.names())
        elif isinstance(idx, dict) and "indexes" in idx:
            names = {i.get("name") for i in idx["indexes"]}
        else:
            names = {getattr(i, "name", None) for i in (idx or [])}
        return name in names
    except Exception as e:
        logging.warning(f"인덱스 조회 실패: {e}")
        return False

def _ensure_index(pc: Pinecone, name: str, dimension: int) -> None:
    if _index_exists(pc, name):
        return

    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    pc.create_index(
        name=name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )

    # 인덱스 준비 완료까지 대기
    for _ in range(30):
        try:
            desc = pc.describe_index(name)
            ready = getattr(desc, "status", {}).get("ready", False) or getattr(desc, "ready", False)
            if ready:
                break
        except Exception:
            pass
        time.sleep(2)

    logging.info(f"Pinecone 인덱스 준비: {name}")


def _load_and_split(file_path: str) -> List[Document]:
    """
    HR 정책 문서를 TextLoader + MarkdownHeaderTextSplitter로 로드하는 함수
    
    Args:
        file_path (str): 마크다운 파일 경로
    
    Returns:
        List[Document]: 분할된 문서 청크들
    """
    
    # 1. TextLoader로 문서 로드
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    
    # 2. Document 객체에서 텍스트 내용 추출
    document_text = documents[0].page_content
    
    # 3. HR 문서 구조에 맞는 헤더 정의
    headers_to_split_on = [
        ("#", "문서제목"),          # # 가이다 플레이 스튜디오(GPS) 직원 복지제도 종합 안내서
        ("##", "정책대분류"),       # ## 1. 휴가 및 휴직 제도
        ("###", "정책세부항목"),    # ### 1.1 연차휴가
        # ("####", "세부절차"),       # #### **신청 절차**
    ]
    
    # 4. MarkdownHeaderTextSplitter로 구조적 분할
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # 헤더 정보 유지 (컨텍스트에 중요)
    )
    
    # 5. 문자열을 split_text에 전달 (Document가 아닌 str)
    md_header_splits = markdown_splitter.split_text(document_text)
    
    # 6. 긴 섹션을 위한 추가 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # HR Q&A에 적합한 크기
        chunk_overlap=200,      # 충분한 컨텍스트 오버랩
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    
    # 7. 최종 분할 적용
    split_docs = text_splitter.split_documents(md_header_splits)
    
    return split_docs


def get_vectorstore(
    index_name: str = "gaida-company-rules",
    file_path: str = "04_복지정책_v1.0.md",
) -> PineconeVectorStore:
    """
    lazy singleton vectorstore factory
    - 인덱스 없으면 생성
    - 파일이 있으면 로드/분할/업서트
    - 결과는 (index_name, abs(file_path)) 키로 캐시
    """
    key: Tuple[str, str] = (index_name, os.path.abspath(file_path))

    with _VSTORE_LOCK:
        if key in _VSTORE_CACHE:
            return _VSTORE_CACHE[key]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    dimension = 1536  # text-embedding-3-small 기준 차원

    pc = _pc()
    _ensure_index(pc, index_name, dimension)

    if os.path.exists(file_path):
        split_docs = _load_and_split(file_path)
        vs = PineconeVectorStore.from_documents(
            documents=split_docs,
            embedding=embeddings,
            index_name=index_name,
        )
    else:
        logging.warning(f"{file_path} 파일이 없어 업서트를 건너뜁니다. 빈 인덱스에서 검색이 수행됩니다.")
        vs = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
        )

    with _VSTORE_LOCK:
        _VSTORE_CACHE[key] = vs
    return vs

def get_retriever(
    index_name: str = "gaida-company-rules",
    file_path: str = "04_복지정책_v1.0.md",
    k: int = 5,
) -> VectorStoreRetriever:
    vs = get_vectorstore(index_name=index_name, file_path=file_path)
    return vs.as_retriever(search_kwargs={"k": k})
