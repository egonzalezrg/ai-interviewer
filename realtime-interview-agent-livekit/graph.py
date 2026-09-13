from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool


load_dotenv(".env.local")


def create_workflow():
    """Create the LangGraph workflow used to run the interview."""

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    #loads the company knowledge base used for RAG.
    pdf_path = os.getenv("COMPANY_PDF_PATH", "./company_profile.pdf")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}. "
            "Please set COMPANY_PDF_PATH or place company_profile.pdf "
            "in the current directory."
        )

    pages = PyPDFLoader(pdf_path).load()

    #splits the document into overlapping chunks for semantic retrieval.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    pages_split = text_splitter.split_documents(pages)

    #stores document embeddings in Chroma for company information searches.
    persist_directory = os.getenv("CHROMA_DIR", "./chroma_store")
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="company_info",
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    @tool
    def company_info_tool(query: str) -> str:
        """Retrieve company information relevant to a candidate's question."""

        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found in the company documents."

        result_parts = []

        for i, doc in enumerate(docs):
            info_number = i + 1
            content = doc.page_content
            formatted_info = f"Info {info_number}:\n{content}"
            result_parts.append(formatted_info)

        return "\n\n".join(result_parts)

    @tool
    def record_answer_tool(answer: str) -> str:
        """Save a candidate's interview answer for later review."""

        with open("interview_answers.txt", "a", encoding="utf-8") as f:
            f.write(f"\nAnswer:\n{answer}\n")
            f.write("-" * 50 + "\n")

        print(f"Recorded answer: {answer[:50]}...")
        return "Answer recorded successfully!"

    tools = [
        company_info_tool,
        record_answer_tool,
    ]

    llm = llm.bind_tools(tools)

    class InterviewState(TypedDict):
        """State passed between nodes in the interview graph."""

        messages: Annotated[Sequence[BaseMessage], add_messages]

    def decide_next_action(state: InterviewState) -> str:
        """Route LLM tool calls to the tool executor."""

        last_message = state["messages"][-1]

        if (
            hasattr(last_message, "tool_calls")
            and last_message.tool_calls
            and len(last_message.tool_calls) > 0
        ):
            return "tool_executor"

        return "end"

    def call_llm(state: InterviewState) -> InterviewState:
        """Generate the interviewer's next response."""

        system_prompt = (
            "You are a professional interviewer conducting a job interview. "
            "You will ask structured questions in this order:\n"
            "1. First: 'Hello! Thank you for joining us today. To get started, "
            "could you tell me a little about yourself and your background?'\n"
            "2. After they respond: 'That's great to hear! Now, I'd love to learn "
            "about your technical background. Could you tell me about your experience "
            "with technology? What technologies, programming languages, or technical "
            "projects have you worked with?'\n"
            "3. After they respond: 'Excellent! Now, I'd like to hear about a time "
            "when you faced a significant challenge, either technical or professional. "
            "Could you walk me through the situation, what obstacles you encountered, "
            "and how you overcame them? What did you learn from that experience?'\n"
            "4. After they respond: 'Thank you for sharing that with me. Now, I'd "
            "like to give you the opportunity to ask me anything about our company, "
            "the role, or anything else you'd like to know. What questions do you "
            "have for me?'\n\n"

            "IMPORTANT ROUTING RULES:\n"
            "- When the candidate asks questions about the company (mission, culture, "
            "revenue, etc.), use the company_info_tool to find relevant information\n"
            "- When the candidate gives answers to your interview questions, use the "
            "record_answer_tool to record their response, then acknowledge it and ask "
            "the next question\n\n"

            "IMPORTANT VOICE RESPONSE RULES:\n"
            "- This is a live spoken interview, so keep responses concise and natural\n"
            "- For simple factual questions, answer in 1-2 short sentences\n"
            "- When company_info_tool returns information, use it only as reference "
            "material to answer the candidate's specific question\n"
            "- NEVER read the retrieved document text word-for-word\n"
            "- NEVER read document headings, page numbers, footers, labels such as "
            "'Info 1', or unrelated sections\n"
            "- Do not summarize the entire retrieved result\n"
            "- Only include information directly relevant to the candidate's question\n"
            "- Do not mention the knowledge base, retrieval system, document, or tool\n"
            "- After answering a company question, stop speaking and allow the "
            "candidate to respond\n"
        )

        messages = [
            SystemMessage(content=system_prompt)
        ] + list(state["messages"])

        message = llm.invoke(messages)

        
        #only send completed conversational responses to the voice layer.
        #tool-call messages remain internal to the LangGraph workflow.
        if not getattr(message, "tool_calls", None):
            writer = get_stream_writer()
            writer({"content": message.content})

        return {"messages": [message]}

    def tool_executor(state: InterviewState) -> InterviewState:
        """Execute tools requested by the interview LLM."""

        tool_calls = state["messages"][-1].tool_calls
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})

            print(f"Running tool: {tool_name}")

            if tool_name == "company_info_tool":
                result = company_info_tool.invoke(tool_args)

            elif tool_name == "record_answer_tool":
                result = record_answer_tool.invoke(tool_args)

            else:
                result = f"Unknown tool: {tool_name}"

            tool_message = ToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_name,
                content=str(result),
            )

            results.append(tool_message)

        print("All tools finished running.")

        return {"messages": results}

    #build the interview graph and connect its execution paths.
    graph = StateGraph(InterviewState)

    graph.add_node("llm", call_llm)
    graph.add_node("tool_executor", tool_executor)

    graph.set_entry_point("llm")

    graph.add_conditional_edges(
        "llm",
        decide_next_action,
        {
            "tool_executor": "tool_executor",
            "end": END,
        },
    )

    graph.add_edge("tool_executor", "llm")

    return graph.compile()