# mvp_tipayo1_fixed.py
# 원본 코드는 변경하지 않고, 이해를 돕기 위한 주석을 추가했습니다.

# ==== 1. 환경변수 로드 ====
from dotenv import load_dotenv
load_dotenv()  # .env 파일에서 환경변수 읽어오기

# ==== 2. 필요한 라이브러리 임포트 ====
from langgraph.graph import MessagesState
from typing_extensions import Any, TypedDict, List, Annotated
from langchain_core.documents import Document
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
import os

# ==== 3. LLM 모델명 수정 ====
llm = ChatOpenAI(model='gpt-4.1', temperature=0)  

# ==== 4. State 클래스들 정의 ====
class State(MessagesState):
    # MessagesState 기반, 대화 상태에 필요한 필드 추가
    question: str     # 사용자가 입력한 질문
    dataset: Any      # 데이터셋(문서) 저장용
    status: bool      # 처리 상태 플래그
    result: str       # 임시 결과 텍스트
    answer: str       # 최종 답변

class Search(TypedDict):
    """가이다 플레이 스튜디오 복지정책 벡터 검색을 위한 쿼리 변환"""
    query: Annotated[str, ..., '''
    사용자의 복지 관련 질문을 가이다 플레이 스튜디오 복지정책 문서에서 검색 가능한 최적화된 쿼리로 변환합니다.
    
    변환 규칙:
    1. 핵심 복지 키워드 추출 및 표준화:
       - 휴가 관련: "연차", "월차", "병가", "가족돌봄휴가", "유급", "무급", "휴가신청"
       - 지원제도: "복지포인트", "교육비", "건강검진", "육아지원금", "출산준비"
       - 업무지원: "장비지원", "노트북", "모니터", "개인장비", "200만원"
       - 절차관련: "전자결재", "팀장승인", "COO승인", "Slack", "구글폼", "HR"
       
    2. 정책 카테고리 명시:
       - "휴가제도", "복리후생", "업무지원", "임신출산", "건강관리", "교육지원"
       
    3. 수치 및 조건 정보 포함:
       - "연 15일", "월 30만원", "연간 300만원", "50만원 한도", "2년 1회"
       - "3년 이상", "만 8세 이하", "80% 이상 출근율"
       
    4. 구어체를 공식 용어로 변환:
       - "언제까지" → "기한 만료일 사용기간"
       - "얼마나" → "금액 한도 지원범위"
       - "어떻게 신청" → "신청절차 승인절차"
       
    5. 동의어 및 관련 용어 확장:
       - "휴가" → "연차 월차 병가 휴직"
       - "지원금" → "복지포인트 교육비 육아지원금"
       - "신청" → "전자결재 승인절차 구글폼"
    
    변환 예시:
    "연차는 언제까지 써야 하나요?" → "연차휴가 사용기한 만료일 12월 31일 미사용 소멸"
    "교육비 지원받을 수 있나요?" → "교육비지원 연간 50만원 한도 자기계발 사전승인"
    "출산 관련 혜택이 뭐가 있어요?" → "출산준비 기프트 육아지원금 태아검진 유급반차"
    ''']

class MyState(TypedDict):
    # LangGraph 파이프라인에 전달될 상태 구조
    question: str
    query: Search
    context: List[Document]
    answer: str

# ==== 5. analyze_query 노드 ====
def analyze_query(state: MyState):
    """
    사용자 질문을 Search TypedDict 구조에 맞춰 최적화된 쿼리로 변환
    """
    s_llm = llm.with_structured_output(Search)  # 구조화된 출력 요청
    query = s_llm.invoke(state['question'])
    return {'query': query}

# ==== 6. 문서 로드 및 벡터스토어 생성 ====
def setup_vectorstore():
    """
    문서를 로드하고 Pinecone 벡터스토어를 설정하는 함수
    - 파일 없을 경우 샘플 데이터 생성
    """
    try:
        file_path = "데이터셋.md"
        if not os.path.exists(file_path):
            # 파일이 없으면 샘플 데이터를 생성
            print(f"경고: {file_path} 파일이 존재하지 않습니다.")
            print("샘플 데이터를 생성합니다...")
            
            sample_data = """# 가이다 플레이 스튜디오 복지정책

## 휴가제도
- 연차휴가: 연간 15일 제공, 12월 31일까지 사용
- 월차휴가: 월 1회 제공
- 병가: 유급 3일, 무급 연장 가능

## 복리후생
- 복지포인트: 월 30만원 제공
- 교육비지원: 연간 50만원 한도
- 건강검진: 2년 1회 전액 지원

## 업무지원
- 장비지원: 노트북, 모니터 등 개인장비 200만원 한도
- 재택근무: 주 2회 가능

## 임신출산지원
- 출산준비 기프트 제공
- 육아지원금: 만 8세 이하 자녀 대상
- 태아검진 유급반차 제공
"""
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(sample_data)
            print(f"{file_path} 파일을 생성했습니다.")

        # 텍스트 파일에서 문서 로드
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        
        # 문서 분할 (chunk_size=1000, overlap=200)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_docs = splitter.split_documents(docs)
        
        # 임베딩 모델 설정
        embedding = OpenAIEmbeddings(model='text-embedding-3-small')
        
        # Pinecone 인덱스명 (공백 제거 및 하이픈 사용)
        index_name = 'gaida-company-rules'
        
        # Pinecone 벡터스토어 생성
        vectorstore = PineconeVectorStore.from_documents(
            splitted_docs, 
            index_name=index_name, 
            embedding=embedding 
        )
        
        return vectorstore

    except Exception as e:
        # 오류 발생 시 상세 안내
        print(f"벡터스토어 설정 중 오류 발생: {e}")
        print("Pinecone 설정을 확인해주세요:")
        print("1. PINECONE_API_KEY가 환경변수에 설정되어 있는지")
        print("2. Pinecone 인덱스가 생성되어 있는지")
        return None

# ==== 7. retrieve 함수 수정 ====
def retrieve(query: str, vectorstore=None):
    """
    벡터스토어를 이용해 유사 문서 검색
    Args:
        query: 최적화된 쿼리 문자열
        vectorstore: PineconeVectorStore 객체
    Returns:
        result_text: 검색 결과 텍스트
        docs: 원본 Document 리스트
    """
    if vectorstore is None:
        print("벡터스토어가 설정되지 않았습니다.")
        return "벡터스토어 오류", []
    
    try:
        # 상위 k=3개 문서 검색
        docs = vectorstore.similarity_search(query, k=3)
        
        # 메타데이터와 내용을 합쳐서 출력용 텍스트 생성
        result_text = '\n\n'.join(
            (f'Source: {doc.metadata}\nContent: {doc.page_content}')
            for doc in docs
        )
        return result_text, docs

    except Exception as e:
        print(f"검색 중 오류 발생: {e}")
        return f"검색 오류: {e}", []

# ==== 8. 메인 실행 코드 ====
if __name__ == "__main__":
    # 벡터스토어 설정
    print("벡터스토어를 설정하고 있습니다...")
    vectorstore = setup_vectorstore()
    
    if vectorstore is not None:
        print("벡터스토어 설정이 완료되었습니다!")
        
        # 테스트 쿼리 실행 예시
        test_query = "연차휴가는 언제까지 사용해야 하나요?"
        result, docs = retrieve(test_query, vectorstore)
        print(f"\n테스트 쿼리: {test_query}")
        print(f"검색 결과: {result}")
    else:
        print("벡터스토어 설정에 실패했습니다.")
