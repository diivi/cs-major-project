from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import chainlit as cl
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import PyPDF2

client = AsyncOpenAI()
cl.instrument_openai()
settings = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Explain a privacy policy",
            message="Explain the privacy policy of facebook.com to me in simple terms. I'm not sure what it means.",
            icon="/public/contract.svg",
        ),
        cl.Starter(
            label="Be my lawyer",
            message="I need help writing a contract for a freelance project. Can you help me draft one?",
            icon="/public/law.svg",
        ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
            icon="/public/terminal.svg",
        ),
        cl.Starter(
            label="Text inviting friend to wedding",
            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
            icon="/public/write.svg",
        ),
    ]

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful legal assistant for lawyers and civilians from India."}],
    )

@cl.on_message
async def main(message: cl.Message):
    if message.elements:
        pdf = [file for file in message.elements if "pdf" in file.mime][0]
        pdf = PyPDF2.PdfReader(pdf.path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
        docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len, is_separator_regex=False).create_documents([pdf_text])

        metadatas = [{"source": f"{i}-pl"} for i in range(len(docs))]
        for doc in docs:
            doc.metadata = metadatas.pop(0)
        vector_store.add_documents(documents=docs)
        vector_store.save_local("faiss_index_constitution")

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model="gpt-4o-mini")

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = rag_chain.invoke(message.content)

        # Step 1: Ask GPT-4o to critically analyze the response
        critique_prompt = f"""
        Critically analyze the following response:
        {response}

        Provide concise points for improvement, including:
        - Any unclear or redundant information
        - Ethical biases that should be avoided
        - Additional clarifications that could enhance understanding
        - Suggestions to improve the overall quality
        """
        critique_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": critique_prompt}],
            **settings
        )

        critique_points = critique_response.choices[0].message.content

        # Step 2: Ask GPT-4o to refine the response
        refinement_prompt = f"""
        Based on the original response and the following critique:
        {critique_points}

        Provide a final improved response:
        {response}
        """
        final_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": refinement_prompt}],
            **settings
        )

        final_content = final_response.choices[0].message.content
        final_message = cl.Message(content=final_content)

        await final_message.send()
        return

    # Handle non-PDF messages
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})

    # Step 3: Perform critical analysis and refinement for regular messages
    critique_prompt = f"""
    Critically analyze the following response:
    {msg.content}

    Provide concise points for improvement, including:
    - Any unclear or redundant information
    - Ethical biases that should be avoided
    - Additional clarifications that could enhance understanding
    - Suggestions to improve the overall quality
    """
    critique_response = await client.chat.completions.create(
        messages=[{"role": "user", "content": critique_prompt}],
        **settings
    )

    critique_points = critique_response.choices[0].message.content

    refinement_prompt = f"""
    Based on the original response and the following critique:
    {critique_points}

    Provide a final improved response:
    {msg.content}
    """
    final_response = await client.chat.completions.create(
        messages=[{"role": "user", "content": refinement_prompt}],
        **settings
    )

    final_content = final_response.choices[0].message.content
    msg.content = final_content
    await msg.update()
