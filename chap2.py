import os
import azure.search.documents
azure.search.documents.__version__

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient, SearchIndexingBufferedSender
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryCaptionResult,
    QueryAnswerResult,
    SemanticErrorMode,
    SemanticErrorReason,
    SemanticSearchResultsType,
    QueryType,
    VectorizedQuery,
    VectorQuery,
    VectorFilterMode,
)
import openai
from dotenv import load_dotenv


load_dotenv()

# =================================
# 0. 接続設定
# =================================

# Azure AI Search
search_service_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
search_service_admin_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
index_name = "gptkbindex"
credential = AzureKeyCredential(search_service_admin_key)
model = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")

# AzureOpenAI

# from openai.types.chat import (
#     ChatCompletion,
#     ChatCompletionChunk
# )

from tenacity import retry, wait_random_exponential, stop_after_attempt

# openai_client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version="2024-07-01-preview",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # 例: https://<resource-name>.openai.azure.com/
openai.api_version = "2024-07-01-preview"                     # リソース作成時に指定したバージョン
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# def generate_embeddings(text, model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")):
#     return openai_client.embeddings.create(input = [text], model=model).data[0].embedding
def generate_embeddings(text, model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")):
    response = openai.Embedding.create(
        model=model,
        input=[text]
    )
    return response["data"][0]["embedding"]

# =================================
# 1. 検索クエリ生成
# =================================

query_prompt_template = """
以下は、過去の会話の履歴と、日本史に関するナレッジベースを検索して回答する必要のあるユーザーからの新しい質問です。
会話と新しい質問に基づいて、検索クエリを作成してください。
検索クエリには、引用されたファイルや文書の名前（例:info.txtやdoc.pdf）を含めないでください。
検索クエリには、括弧 []または<<>>内のテキストを含めないでください。
検索クエリを生成できない場合は、数字 0 だけを返してください。
"""

messages = [{'role': 'system', 'content': query_prompt_template}]

query_prompt_few_shots = [
    {'role': 'user', 'content': '徳川家康ってなにした人'},
    {'role': 'assistant', 'content': '徳川家康 人物 歴史'},
    {'role': 'user', 'content': '徳川家康の武功を教えてください。'},
    {'role': 'assistant', 'content': '徳川家康 人物 武功 業績'}
]

for shot in query_prompt_few_shots:
    messages.append({'role': shot.get('role'), 'content':shot.get('content')})

user_q = "源実朝ってどんな人"
messages.append({'role': 'user', 'content': user_q})

messages

completion = openai.ChatCompletion.create(
    messages=messages,
    engine=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temepature=0.0,
    max_tokens=100,
    n=1
)

query_text = completion.choices[0].message.content
print(query_text)

# ===========================================
# 2. 検索インデックスから関連文書を取得(Retrieve)
# ===========================================
def nonewlines(s: str) -> str:
    return s.replace('\n', ' ').replace('\r', ' ').replace('[', ' [').replace(']', ' ]')

search_client = SearchClient(search_service_endpoint, index_name, credential=credential)
docs = search_client.search(
    search_text=query_text,
    filter=None,
    top=3,
    vector_queries=[VectorizedQuery(vector=generate_embeddings(query_text), k_nearest_neighbors=3, fields="embedding")]
)

results = ["SOURCE" + doc['sourcepage'] + ":" + nonewlines(doc['content']) for doc in docs]
print(results)

# ===========================================
# 3. ChatGPTを利用した回答の生成
# ===========================================
system_message_chat_conversation = """
日本の鎌倉時代の歴史に関する読解問題に答えるアシスタントです。
If you cannot guess the answer to a question from the SOURCE, answer "I don't know".
Answers must be in Japanese.

# Restrictions
- The SOURCE prefix has a colon and actual information after the filename, and each fact used in the response must include the name of the source.
- To reference a source, use a square bracket. For example, [info1.txt]. Do not combine sources, but list each source separately. For example, [info1.txt][info2.pdf].
"""

messages = [{'role': 'system', 'content': system_message_chat_conversation}]

user_q =  "源実朝ってどんな人"
context = "\n".join(results)
messages.append({'role': 'user', 'content': user_q + '\n\n' + context})

messages

chat_coroutine = openai.Chatcompletion.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    messages=messages,
    temepature = 0.0,
    max_tokens=1024,
    n=1,
    stream=False
)

print(chat_coroutine.choices[0].message.content)