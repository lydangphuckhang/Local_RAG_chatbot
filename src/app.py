
import os
os.environ["HF_HOME"] = "D:/Project/Local_RAG_chatbot/models_cache"

import chainlit as cl

from rag_pipeline import create_rag_chain


@cl.on_chat_start
async def on_chat_start():

    await cl.Message(
        content="Welcome to the Local RAG Chatbot!\n"
                "The system is initializing..."
    ).send()

    chain = create_rag_chain()

    cl.user_session.set("chain", chain)
    cl.user_session.set("chat_history", [])

    await cl.Message(
        content="System initialized. What would you like to ask?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):

    chain = cl.user_session.get("chain")
    chat_history = cl.user_session.get("chat_history")

    msg = cl.Message(content="")
    await msg.send()

    response = await chain.ainvoke({
        "question": message.content,
        "chat_history": chat_history
    })

    answer = response.get("answer", "No answer generated.")
    source_documents = response.get("source_documents", [])

    text_elements = []

    if source_documents:

        for source_idx, source_doc in enumerate(source_documents):

            source_name = f"Source {source_idx + 1}"

            text_elements.append(
                cl.Text(
                    content=source_doc.page_content,
                    name=source_name
                )
            )

        source_names = [el.name for el in text_elements]

        answer += f"\n\n**References:** {', '.join(source_names)}"

    msg.content = answer
    msg.elements = text_elements

    await msg.update()

    # Update conversation history
    chat_history.append((message.content, answer))

    cl.user_session.set("chat_history", chat_history)
