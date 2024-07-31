import os
import streamlit as st
from tavily import TavilyClient
from pprint import pprint
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langchain_openai import OpenAIEmbeddings
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

tavily = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)

st.set_page_config(page_title="RAG Agent", page_icon="🤖", layout="wide")

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]


def main():
    llm_model = st.sidebar.selectbox(
        "Select Model",
        options=[
            "GPT-4o-Mini",
        ]
    )

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    retriever = vectorstore.as_retriever()


    # RAG 에이전트 노드 및 엣지 정의
    def retrieve(state):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.invoke(question)
        print(question)
        print(documents)
        return {"documents": documents, "question": question}

    def generate(state):
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        sources = [d.metadata.get("source") for d in documents]

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question}) + "\nsource:\n" + "\n".join(
            set(sources))
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                continue

        if len(filtered_docs) == 0:
            web_search = "Yes"

        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def web_search(state):
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """

        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = tavily.search(query=question)['results']

        web_results = [Document(page_content=d["content"], metadata={'title': d["title"], 'source': d["url"]}) for d in
                       docs]
        return {"documents": web_results, "question": question}

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        web_search = state["web_search"]
        state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def check_hallucination(state):
        """
        Determines whether the generation is grounded in the document.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            return "supported"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    # RAG 에이전트 그래프 구성
    workflow = StateGraph(GraphState)
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "grade_documents")
    workflow.add_conditional_edges(
        "generate",
        check_hallucination,
        {
            "supported": END,
            "not supported": "generate",
        },
    )

    app = workflow.compile()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # rag_chain 정의
    rag_system = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise"""

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rag_system),
            ("human", "question: {question}\n\n context: {context} "),
        ]
    )

    # Chain
    rag_chain = rag_prompt | llm | StrOutputParser()

    # retrieval_grader 정의
    retrieval_grader_system = """You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """

    retrieval_grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", retrieval_grader_system),
            ("human", "question: {question}\n\n document: {document} "),
        ]
    )

    retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

    # hallucination_grader 정의
    hallucination_grader_system = """You are a grader assessing whether
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
        single key 'score' and no preamble or explanation."""

    hallucination_grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", hallucination_grader_system),
            ("human", "documents: {documents}\n\n answer: {generation} "),
        ]
    )

    hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

    # ----------------------------------------------------------------------
    # Streamlit 앱 UI
    st.title("RAG Agent by OpenAI")

    input_topic = st.text_input(
        ":female-scientist: Enter a topic",
        value="agent memory",
    )

    generate_report = st.button("Generate Report")

    if generate_report:
        with st.spinner("Generating Report"):
            inputs = {"question": input_topic}
            for output in app.stream(inputs):
                for key, value in output.items():
                    print(f"Finished running: {key}:")
            final_report = value["generation"]
            st.markdown(final_report)

    st.sidebar.markdown("---")
    if st.sidebar.button("Restart"):
        st.session_state.clear()
        st.experimental_rerun()


main()
# 실행 방법: 터미널에 streamlit run Day3_RAG_agent_streamlit.py 입력