import os
import azure.search.documents
import langchain
import csv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery
)

# from langchain.agents import (
#     AgentType,
#     Tool,
#     initialized_agent
# )
# from langchain.agents.mrkl import prompt
from langchain.agents import (
    AgentType,
    Tool,
    initialize_agent
)

from langchain.chat_models import AzureChatOpenAI
from langchain.tools import BaseTool
from typing import Dict
from pydantic import Field

import openai
from dotenv import load_dotenv

load_dotenv()
# =================================
# 0. 接続設定
# =================================

# Azure AI Search
search_service_endpoint: str = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
search_service_admin_key: str = os.getenv("AZURE_AI_SEARCH_API_KEY")
search_query_key: str = os.getenv("AZURE_AI_SEARCH_QUERY_KEY")
index_name = "gptkbindex"
credential = AzureKeyCredential(search_query_key)

# AzureOpenAI
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # 例: https://<resource-name>.openai.azure.com/
openai.api_version = "2024-07-01-preview"                     # リソース作成時に指定したバージョン
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

from tenacity import retry, wait_random_exponential, stop_after_attempt

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_embeddings(text, model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")):
    response = openai.Embedding.create(
        model=model,
        input=[text]
    )
    return response["data"][0]["embedding"]

# =================================
# 1.武将検索ツールのRetrieve 実装
# =================================
def retrieve(querytext: str):
    search_client = SearchClient(search_service_endpoint, index_name, credential=credential)
    docs = search_client.search(
        search_text=query_text,
        filter=None,
        top=3,
        vector_queries=[VectorizedQuery(vector=generate_embeddings(query_text), k_nearest_neighbors=3, fields="embedding")]
    )
    results = [doc['sourcepage'] + ":" + nonewlines(doc['content']) for doc in docs]
    content = "\n".join(results)
    return content

def nonewlines(s: str) -> str:
    return s.replace('\n', ' ').replace('\r', ' ').replace('[', ' [').replace(']', '] ')


# =================================
# 2.カフェ検索ツールの定義
# =================================
class CafeSearchTool(BaseTool):
    data: dict[str, str] = Field(default_factory=dict)
    name = "CafeSearchTool"
    description: str = "武将のゆかりのカフェを検索するのに便利です。カフェの検索クエリには、武将の**名前のみ**を入力してください。"

    def _run(self, query:str) -> str:
        filename = "data/restaurantinfo.csv"
        key_field = "name"
        try:
            with open(filename, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.data[row[key_field]] = "\n".join([f"{i}:{row[i]}" for i in row])
        except Exception as e:
            print("File read error:", e)

        return self.data.get(query, "")
    
# =================================
# 3.Toolの定義
# =================================
tools = [
    Tool(name="PeopleSearchTool",
        func=retrieve,
        coroutine=retrieve,
        description="日本の歴史の人物情報の検索に便利です。ユーザーの質問から検索クエリーを生成して検索します。クエリーは文字列のみを受け付けます"
        ),
    CafeSearchTool()
]

# =================================
# 4.LLMの定義
# =================================
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.0,
    )

q = "鎌倉幕府第二代征夷大将軍の名前とその将軍にゆかりの地にあるカフェの名前を教えて"

# =================================
# 5.Agentの実行
# =================================
SUFFIX = """
Answer should be in Japanese.
"""
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs=dict(suffix=SUFFIX + prompt.SUFFIX),
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate"
)

result = agent_chain.run(q)
result