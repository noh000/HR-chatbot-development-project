# db.py

from dotenv import load_dotenv
import os
import logging
import threading
import time
from typing import List, Dict, Tuple

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

# --- 초기 설정 ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOCS_DIRECTORY = "."  # db.py가 있는 폴더와 다른 위치에 문서가 있다면 변경
HR_DOCUMENT_FILES = [
    "04_복지정책_v1.0.md"
]
# HR_DOCUMENT_FILES = [                 # 문서 추가 후, 변경
#     "01_직원핸드북_v1.0_2025-01-10.md",
#     "02_근무정책_v1.0.md",
#     "03_휴가정책_v1.0.md",
#     "04_복지정책_v1.0.md",
#     "05_장비·보안정책_v1.0.md",
# ]

EXISTING_HR_DOCS = [
    os.path.join(DOCS_DIRECTORY, f) for f in HR_DOCUMENT_FILES 
    if os.path.exists(os.path.join(DOCS_DIRECTORY, f))
]

# --- 전역 변수 및 캐시 ---
_VSTORE_CACHE: Dict[str, PineconeVectorStore] = {}
_VSTORE_LOCK = threading.Lock()


# --- Pinecone 클라이언트 관리 ---
def _get_pinecone_client() -> Pinecone:
    """Pinecone 클라이언트를 안전하게 초기화하고 반환합니다."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY가 환경 변수에 설정되지 않았습니다.")
    return Pinecone(api_key=api_key)

def _index_exists(pc: Pinecone, name: str) -> bool:
    """Pinecone 인덱스 존재 여부를 견고하게 확인합니다."""
    try:
        indexes_info = pc.list_indexes()
        # list_indexes()의 반환 타입에 따라 유연하게 처리
        if hasattr(indexes_info, 'names'): # v3.x
            return name in indexes_info.names()
        elif isinstance(indexes_info, list): # v2.x
            return name in [idx['name'] for idx in indexes_info]
        return False
    except Exception as e:
        logging.warning(f"인덱스 조회 중 오류 발생: {e}")
        return False

def _ensure_index(pc: Pinecone, name: str, dimension: int):
    """Pinecone 인덱스가 없으면 생성하고 준비될 때까지 대기합니다."""
    if _index_exists(pc, name):
        logging.info(f"Pinecone 인덱스 '{name}'가 이미 존재합니다.")
        return

    logging.info(f"Pinecone 인덱스 '{name}'를 생성합니다.")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    try:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # 인덱스가 준비될 때까지 최대 60초 대기
        for _ in range(30):
            status = pc.describe_index(name).status
            if status and status.get('ready'):
                logging.info(f"Pinecone 인덱스 '{name}' 준비 완료.")
                return
            time.sleep(2)
        logging.warning(f"'{name}' 인덱스가 시간 내에 준비되지 않았습니다.")
    except Exception as e:
        logging.error(f"'{name}' 인덱스 생성 실패: {e}")
        raise

# --- 문서 처리 ---
def _load_and_split_docs(file_paths: List[str]) -> List[Document]:
    """여러 마크다운 파일을 로드하고 구조적으로 분할하여 문서 청크 리스트를 반환합니다."""
    all_splits = []
    headers_to_split_on = [
        ("#", "doc_title"),
        ("##", "main_category"),
        ("###", "sub_category"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False  # 헤더 정보 유지 (컨텍스트에 중요)
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )

    for file_path in file_paths:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            if not documents:
                logging.warning(f"'{file_path}' 파일이 비어있습니다.")
                continue
            
            md_splits = markdown_splitter.split_text(documents[0].page_content)
            
            # 각 분할된 문서에 파일 출처(source) 메타데이터 추가
            for doc in md_splits:
                doc.metadata["source"] = os.path.basename(file_path)

            splits = text_splitter.split_documents(md_splits)
            all_splits.extend(splits)
            logging.info(f"'{file_path}' 로드 및 분할 완료: {len(splits)}개 청크 생성.")
        except Exception as e:
            logging.error(f"'{file_path}' 처리 중 오류 발생: {e}")
            
    return all_splits

# --- VectorStore ---
def get_vectorstore(
    index_name: str = "gaida-hr-rules",
    recreate: bool = False,
) -> PineconeVectorStore:
    """
    Pinecone 벡터 저장소를 가져오거나 생성합니다.
    - 캐시된 인스턴스가 있으면 반환합니다.
    - recreate=True이면, 인덱스 내 문서를 모두 삭제하고 새로 업로드합니다.
    - DB가 비어있으면 자동으로 문서를 업로드합니다.
    """
    with _VSTORE_LOCK:
        if not recreate and index_name in _VSTORE_CACHE:
            logging.info(f"캐시된 VectorStore 인스턴스 '{index_name}'를 반환합니다.")
            return _VSTORE_CACHE[index_name]
    """
    임베딩 모델별 dimension
    OpenAI text-embedding-ada-002: 1536
    OpenAI text-embedding-3-small: 1536
    OpenAI text-embedding-3-large: 3072
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    dimension = 1536

    pc = _get_pinecone_client()
    _ensure_index(pc, index_name, dimension)
    index = pc.Index(index_name)

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    stats = index.describe_index_stats()
    vector_count = stats.get("total_vector_count", 0)

    # 문서 업로드 조건: recreate=True 이거나, DB가 비어있을 때
    if recreate or vector_count == 0:
        if recreate and vector_count > 0:
            logging.info(f"인덱스 '{index_name}'의 모든 벡터({vector_count}개)를 삭제합니다.")
            index.delete(delete_all=True)
        
        if EXISTING_HR_DOCS:
            logging.info(f"존재하는 문서 파일을 DB에 업로드합니다: {EXISTING_HR_DOCS}")
            split_docs = _load_and_split_docs(EXISTING_HR_DOCS)
            if split_docs:
                logging.info(f"총 {len(split_docs)}개 청크를 인덱스 '{index_name}'에 업로드합니다.")
                vectorstore.add_documents(documents=split_docs, batch_size=100)
        else:
            logging.warning("존재하는 HR 문서 파일이 없어 업로드를 건너뜁니다.")
    else:
        logging.info(f"인덱스 '{index_name}'에 {vector_count}개의 벡터가 이미 존재합니다. (재생성 원할 시 recreate=True)")

    with _VSTORE_LOCK:
        _VSTORE_CACHE[index_name] = vectorstore
    return vectorstore

if __name__ == "__main__":
    # 스크립트를 직접 실행할 때 벡터 저장소를 생성하고 문서를 업로드합니다.
    # recreate=True로 설정하면 실행 시마다 기존 문서를 모두 지우고 새로 업로드합니다.
    logging.info("스크립트를 직접 실행하여 Pinecone 벡터 저장소 설정을 시작합니다.")
    
    # 처음 생성하거나, 강제로 문서를 다시 업로드하고 싶을 때 recreate=True
    vectorstore = get_vectorstore(recreate=False) 
    
    if vectorstore:
        logging.info("벡터 저장소 설정이 성공적으로 완료되었습니다.")
        # 간단한 테스트 검색을 통해 retriever가 정상 동작하는지 확인
        try:
            retriever = vectorstore.as_retriever()
            test_query = "연차 휴가"
            retrieved_docs = retriever.invoke(test_query)
            if retrieved_docs:
                logging.info(f"테스트 쿼리 '{test_query}'에 대한 검색 성공: {len(retrieved_docs)}개 문서 반환.")
            else:
                logging.warning(f"테스트 쿼리 '{test_query}'에 대한 검색 결과가 없습니다. 인덱스는 생성되었으나 문서가 비어있을 수 있습니다.")
        except Exception as e:
            logging.error(f"테스트 검색 중 오류 발생: {e}")
    else:
        logging.error("벡터 저장소 설정에 실패했습니다.")