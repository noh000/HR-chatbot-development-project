import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# LangSmith 설정 (환경변수에서 자동 인식됨)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "hhayan-test"

# =========================
# 1. 환경 변수 로드
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =========================
# 2. 문서 로드
# =========================
loader = TextLoader("./docs/sample.txt", encoding="utf-8")
documents = loader.load()

# =========================
# 3. Split (chunk 단위 쪼개기)
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

print("=== Split 후 문서 확인 ===")
for i, d in enumerate(docs[:3]):  # 앞부분 몇개만 확인
    print(f"[chunk {i}]")
    print(d.page_content[:200], "\n---")

# =========================
# 4. 벡터스토어 생성
# =========================
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(docs, embeddings)

# =========================
# 5. LLM + RAG 체인 구성
# =========================
llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# =========================
# 6. 테스트 질문 (입력 받기)
# =========================
query = input("질문을 입력하세요: ")
answer = qa_chain.run(query)

print("\n=== 최종 답변 ===")
print(answer)
