import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END


# =========================
# 1. 환경 변수 로드
# =========================
load_dotenv()


# =========================
# 2. State 정의
# =========================
class State(TypedDict):
    query: str
    docs: List[str]
    answer: str


# =========================
# 3. 문서 로딩 + Split + VectorStore
# =========================
def load_and_split_docs() -> FAISS:
    loader = TextLoader("./docs/sample.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    print("=== Split 후 문서 확인 ===")
    for i, d in enumerate(docs[:3]):
        print(f"[chunk {i}]\n{d.page_content}\n---")

    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.from_documents(docs, embeddings)


# =========================
# 4. Node: 검색
# =========================
def retrieve_node(state: State):
    vectorstore = load_and_split_docs()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.get_relevant_documents(state["query"])
    return {"docs": [r.page_content for r in results]}


# =========================
# 5. Node: 답변 생성
# =========================
def generate_answer_node(state: State):
    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    context = "\n\n".join(state["docs"])
    prompt = f"다음 문서를 참고하여 질문에 답하세요.\n\n문서:\n{context}\n\n질문: {state['query']}"
    resp = llm.invoke(prompt)
    return {"answer": resp.content}


# =========================
# 6. LangGraph 구성
# =========================
builder = StateGraph(State)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate_answer", generate_answer_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()
