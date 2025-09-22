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
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        logging.warning("PINECONE_API_KEY가 설정되지 않았습니다. Pinecone 호출이 실패할 수 있습니다.")
    return Pinecone(api_key=api_key)

def _index_exists(pc: Pinecone, name: str) -> bool:
    try:
        # SDK 버전에 따른 호환 처리
        idx = pc.list_indexes()
        # v3+: 객체에 .names() 또는 dict 형태가 있을 수 있음
        names = set()
        if hasattr(idx, "names"):
            names = set(idx.names())
        elif isinstance(idx, dict) and "indexes" in idx:
            names = {i.get("name") for i in idx["indexes"]}
        else:
            # 과거/기타 형태
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
    # 인덱스 준비 대기
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
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    base = os.path.basename(file_path)
    for i, d in enumerate(split_docs):
        d.metadata["source"] = f"{base}#chunk-{i}"
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
    dimension = 1536  # text-embedding-3-small

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
