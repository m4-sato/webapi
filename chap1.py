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

from dotenv import load_dotenv


load_dotenv()

# =================================
# 1. 接続設定
# =================================

# Azure AI Search
search_service_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
search_service_admin_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
index_name = "gptkbindex"
credential = AzureKeyCredential(search_service_admin_key)
model = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")

# =================================
# 2. キーワード検索
# =================================
query = "源実友のお歌にはどのような特徴があったのでしょうか？"  
search_client = SearchClient(search_service_endpoint, index_name, credential=credential)
docs = search_client.search(
    search_text=query,
    top=3,
    highlight_fields="content-3",
    select="sourcepage,content,category"
)

for doc in docs:
    print(f"Source: {doc['sourcepage']}")
    print(f"Score: {doc['@search.score']}")
    print(f"Content: {doc['content']}")
    print(f"Category: {doc['category']}\n")


# =================================
# 3. ベクトル類似性検索
# =================================

import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

# client = AzureOpenAI(
#     api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
#     api_version = "2023-05-15",
#     azure_endpoint = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")
# )

# 1) Azure Resource に紐づく情報を設定
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")  # 例: https://<your-resource-name>.openai.azure.com/
openai.api_version = "2023-05-15"  # (エンドポイント作成時に指定したバージョン)
openai.api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY")

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))

def generate_embeddings(text, model=model):
    return openai.Embedding.create(input=[text], model=model).data[0].embedding

query =  "源実友のお歌にはどのような特徴があったのでしょうか？"  

search_client = SearchClient(search_service_endpoint, index_name, credential=credential)
vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=3, fields="embedding")

docs = search_client.search(
    search_text=None,
    vector_queries = [vector_query],
    select = ["sourcepage", "content", "category"]
)

for doc in docs:
    print(f"Source: {doc['sourcepage']}")
    print(f"Score: {doc['@search.score']}")
    print(f"Content: {doc['content']}")
    print(f"Category: {doc['category']}\n")

# ===================== 推論 ===========================
query = "What were the characteristics of Minamoto Sanetomo's poetry?" 

search_client = SearchClient(search_service_endpoint, index_name, credential=credential)
vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=3, fields="embedding")

docs = search_client.search(
    search_text=None,
    vector_queries = [vector_query],
    select = ["sourcepage", "content", "category"]
)

for doc in docs:
    print(f"Source: {doc['sourcepage']}")
    print(f"Score: {doc['@search.score']}")
    print(f"Content: {doc['content']}")
    print(f"Category: {doc['category']}\n")

# =================================
# 4. ハイブリッド検索
# =================================

query = "源実友は征夷大将軍として知られているだけでなく、ある有名な趣味も持っています。それは何ですか。" 

search_client = SearchClient(search_service_endpoint, index_name, credential=credential)
vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=50, fields="embedding")

docs = search_client.search(
    search_text=None,
    vector_queries = [vector_query],
    select = ["sourcepage", "content", "category"],
    top=3
)

for doc in docs:
    print(f"Source: {doc['sourcepage']}")
    print(f"Score: {doc['@search.score']}")
    print(f"Content: {doc['content']}")
    print(f"Category: {doc['category']}\n")


# =================================
# 5. セマンティックハイブリッド検索
# =================================

query =  "１３人の合議制に含まれるメンバー一覧"  

search_client = SearchClient(search_service_endpoint, index_name, credential=credential)
vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=50, fields="embedding")

docs = search_client.search(
    search_text=query,
    vector_queries = [vector_query],
    select = ["sourcepage", "content", "category"],
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name = "default",
    query_caption=QueryCaptionType.EXTRACTIVE,
    query_answer=QueryAnswerType.EXTRACTIVE,
    top=3
)

semantic_answers = docs.get_answers()
for answer in semantic_answers:
    if answer.highlights:
        print(f"Semantic Answer: {answer.highlights}")
    else:
        print(f"Sementic Answer: {answer.text}")
    print(f"Semantic Answer Score: {answer.score}\n")

for doc in docs:
    print(f"Source: {doc['sourcepage']}")
    print(f"Score: {doc['@search.score']}")
    print(f"Content: {doc['content']}")
    print(f"Category: {doc['category']}\n")

    captions = doc["@search.captions"]
    if captions:
        caption = caption[0]
        if caption.highlights:
            print(f"Caption: {caption.highlights}\n")
        else:
            print(f"Caption: {caption.text}\n")

# ================================================
# 5. セマンティックアンサーとセマンティックキャプション
# ================================================

query =  "源頼朝が征夷大将軍に任命されたのはいつ"  

search_client = SearchClient(search_service_endpoint, index_name, credential=credential)
vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=50, fields="embedding")

docs = search_client.search(
    search_text=query,
    vector_queries = [vector_query],
    select = ["sourcepage", "content", "category"],
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name = "default",
    query_caption=QueryCaptionType.EXTRACTIVE,
    query_answer=QueryAnswerType.EXTRACTIVE,
    top=3
)

semantic_answers = docs.get_answers()
for answer in semantic_answers:
    if answer.highlights:
        print(f"Semantic Answer: {answer.highlights}")
    else:
        print(f"Sementic Answer: {answer.text}")
    print(f"Semantic Answer Score: {answer.score}\n")

for doc in docs:
    print(f"Source: {doc['sourcepage']}")
    print(f"Score: {doc['@search.score']}")
    print(f"Content: {doc['content']}")
    print(f"Category: {doc['category']}\n")

    captions = doc["@search.captions"]
    if captions:
        caption = caption[0]
        if caption.highlights:
            print(f"Caption: {caption.highlights}\n")
        else:
            print(f"Caption: {caption.text}\n")

query =  "守護・地頭を設置した人は誰"  

search_client = SearchClient(search_service_endpoint, index_name, credential=credential)
vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=50, fields="embedding")

docs = search_client.search(
    search_text=query,
    vector_queries = [vector_query],
    select = ["sourcepage", "content", "category"],
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name = "default",
    query_caption=QueryCaptionType.EXTRACTIVE,
    query_answer=QueryAnswerType.EXTRACTIVE,
    top=3
)

semantic_answers = docs.get_answers()
for answer in semantic_answers:
    if answer.highlights:
        print(f"Semantic Answer: {answer.highlights}")
    else:
        print(f"Sementic Answer: {answer.text}")
    print(f"Semantic Answer Score: {answer.score}\n")

for doc in docs:
    print(f"Source: {doc['sourcepage']}")
    print(f"Score: {doc['@search.score']}")
    print(f"Content: {doc['content']}")
    print(f"Category: {doc['category']}\n")

    captions = doc["@search.captions"]
    if captions:
        caption = caption[0]
        if caption.highlights:
            print(f"Caption: {caption.highlights}\n")
        else:
            print(f"Caption: {caption.text}\n")