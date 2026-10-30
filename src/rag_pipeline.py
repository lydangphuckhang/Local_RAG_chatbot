
import os
os.environ["HF_HOME"] = "D:/Project/Local_RAG_chatbot/models_cache"

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

from llm_service import QwenLocal
from retriever import get_retriever
from embedding_service import get_embedding_model


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(chat_history):
    return "\n".join(
        f"Human: {question}\nAssistant: {answer}"
        for question, answer in chat_history
    )


def create_rag_chain():

    db_path = "D:/Project/Local_RAG_chatbot/chroma_db"

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Vector store not found at {db_path}. "
            "Please run ingest_data.py first."
        )

    # Initialize embedding model and vector database
    embedder = get_embedding_model()

    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embedder
    )

    retriever = get_retriever(vector_store, k=2)

    # Initialize LLM
    qwen_service = QwenLocal()
    llm = qwen_service.get_llm()

    # Prompt for question rewriting
    contextualize_q_template = """
    <|im_start|>system
    You are an expert at reformulating follow-up questions.

    Given the conversation history and the latest user question,
    rewrite the latest question into a standalone question that
    can be understood without the previous conversation.

    Rules:
    - Resolve pronouns and references using the chat history.
    - Preserve the original meaning.
    - Do not answer the question.
    - If the question is already independent, keep its meaning unchanged.

    Chat history:
    {chat_history}
    <|im_end|>
    <|im_start|>user
    Latest question: {question}
    <|im_end|>
    <|im_start|>assistant
    """

    contextualize_q_prompt = PromptTemplate.from_template(
        contextualize_q_template
    )

    # Question rewriting chain
    question_generator = (
        {
            "chat_history": lambda x: format_chat_history(x["chat_history"]),
            "question": lambda x: x["question"]
        }
        | contextualize_q_prompt
        | llm
        | StrOutputParser()
    )

    # Answer generation prompt
    answer_template = """
    <|im_start|>system
    You are a helpful and precise AI assistant.

    Use the retrieved context to answer the user's question.

    Use the conversation history only to understand references
    and the context of the conversation.

    If the answer cannot be found in the retrieved context,
    say that you don't know. Do not make up information.

    Previous conversation:
    {chat_history}

    Retrieved context:
    {context}
    <|im_end|>
    <|im_start|>user
    Question: {question}
    <|im_end|>
    <|im_start|>assistant
    """

    answer_prompt = PromptTemplate.from_template(answer_template)

 
    answer_chain = (
        {
            "context": lambda x: format_docs(x["source_documents"]),
            "question": lambda x: x["question"],
            "chat_history": lambda x: format_chat_history(x["chat_history"])
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    # Complete conversational RAG chain
    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=question_generator
        )
        .assign(
            source_documents=lambda x: retriever.invoke(
                x["standalone_question"]
            )
        )
        .assign(
            answer=answer_chain
        )
    )

    return rag_chain


if __name__ == "__main__":

    try:
        chain = create_rag_chain()

        chat_history = []

        questions = [
            "What is INSA Centre Val de Loire?",
            "Where is it located?"
        ]

        for question in questions:

            print(f"\nUser: {question}")

            response = chain.invoke({
                "question": question,
                "chat_history": chat_history
            })

            answer = response["answer"]

            print(f"RAG: {answer}")

            chat_history.append((question, answer))

    except Exception as error:
        print(f"\nPipeline execution failed: {error}")
