import os
import chainlit as cl
# from langchain_community.document_loaders import PyMuPDFLoader
from langchain.document_loaders import PyMuPDFLoader
# from langchain_openai import AzureChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.text_splitter import SpacyTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import shutil
from pydantic import BaseModel

# 環境変数をロード
load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
    chunk_size=2048
    )

# chat = AzureChatOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version="2024-08-01-preview",
#     model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY")
#     )

chat = AzureChatOpenAI(
    # ここに Azure Endpoint, API Key などを指定
    openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

prompt = PromptTemplate(
    template="""文章を元に質問に答えてください。

文章:
{document}

質問: {query}
""",
    input_variables=["document", "query"]
)

database = Chroma(
    persist_directory="./data",
    embedding_function=embeddings
)
text_splitter = SpacyTextSplitter(
    chunk_size=300,
    pipeline="ja_core_news_sm"
)

@cl.on_chat_start
async def on_chat_start():
    files = None

    while files is None:
        files= await cl.AskFileMessage(
            max_size_mb=20,
            content="PDFを選択してください",
            accept = ["application/pdf"],
            raise_on_timeout=False,
            ).send()
    file = files[0]
    
    print("file:", file)
    print("type(file):", type(file))

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    # # ファイル内容を非同期で読み込む
    # file_content = await file.read()
    
    # with open(f"tmp/{file.name}", "wb") as f:
    #     f.write(file.content)
 
    # documents = PyMuPDFLoader(f"tmp/{file.name}").load()
    
    # ファイルを直接読み込む
    shutil.copy(file.path, f"tmp/{file.name}")
    documents = PyMuPDFLoader(file.path).load()
    # 分割されたテキストリストを生成
    splitted_documents = text_splitter.split_documents(documents)

    database = Chroma(
        embedding_function=embeddings,
    )

    database.add_documents(splitted_documents)

    cl.user_session.set(
        "database",
        database
    )


    await cl.Message(content=f"`{file.name}`の読み込みが完了しました。質問を入力してください。").send()



@cl.on_message
async def on_message(input_message):
    print("入力されたメッセージ:" + input_message.content)

    database = cl.user_session.get("database")

    documents = database.similarity_search(input_message.content)

    documents_string = ""

    for document in documents:
        documents_string += f"""
    -----------------------------
    {document.page_content}
    """

    result = chat([
        HumanMessage(content=prompt.format(document=documents_string, query=input_message.content))
    ])

    await cl.Message(content=result.content).send()