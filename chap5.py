import os
# import openai
import langchain
import azure.search.documents
# from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
# from langchain_openai import AzureOpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.vectorstores import AzureSearch
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from dotenv import load_dotenv

load_dotenv()

# =================================
# 0. 接続設定
# =================================

vector_store_address: str = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
# vector_store_password: str = os.getenv("AZURE_AI_SEARCH_QUERY_KEY")
vector_store_password: str = os.getenv("AZURE_AI_SEARCH_API_KEY")
index_name = "gptkbindex"
search_analyzer_name = "ja.lucene"
credential = AzureKeyCredential(vector_store_password)
model: str = "embedding"
index_name: str = "gptkbindex"

embeddings = OpenAIEmbeddings(
    openai_api_base=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),     # 例: "https://<リソース名>.openai.azure.com/"
    openai_api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),       # Azure OpenAI Key
    openai_api_type="azure",                                # Azure を使う場合は必須
    openai_api_version="2023-05-15",                        # Azure OpenAI のバージョン
    model_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),                        # ← azure_deployment=model は使わず、deployment_name に
    chunk_size=1000,
    chunk_overlap=0
)

# embeddings = AzureOpenAIEmbeddings(
#     azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),      # 例: "https://<リソース名>.openai.azure.com"
#     azure_api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),        # Azure OpenAI key
#     azure_api_version="2023-05-15",                                    # Azure 用のバージョン
#     azure_deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
#     model_kwargs={  # chunk_size/chunk_overlapなどはこちらにまとめる
#         "chunk_size": 1000,
#         "chunk_overlap": 0
#     }
# )

# embeddings = AzureOpenAIEmbeddings(
#     azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
#     api_version="2023-05-15",
#     deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
#     chunk_size=1000,
#     chunk_overlap=0
# )

# embeddings = AzureOpenAIEmbeddings(
#     model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
#     openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
#     chunk_size=2048
#     )

# vector_store: AzureSearch = AzureSearch(
#     azure_search_endpoint = vector_store_address,
#     azure_search_key = vector_store_password,
#     index_name = index_name,
#     embedding_function=embeddings.embed_query,
#     semantic_configuration_name="default"
# )
vector_store = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    semantic_configuration_name="default",
    doc_vector_field="embedding",               # ベクトルフィールド名を "embedding" と指定
    vector_dimensions=1536
)

def create_search_index():
    fields = [
        SimpleField(name="id", type="Edm.String", key=True),
        SearchableField(
            name="content", type="Edm.String", analyzer_name=search_analyzer_name
        ),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            hidden=False,
            searchable=True,
            filterable=False,
            sortable=False,
            facetable=False,
            vector_search_dimensions=1536,
            vector_search_profile_name="embedding_config"
        ),
        SimpleField(name="category", type="Edm.String", filterable=True, facetable=True),
        SimpleField(name="sourcepage", type="Edm.String", filterable=True, facetable=True),
        SimpleField(name="sourcefile", type="Edm.String", filterable=True, facetable=True),
        SimpleField(name="metadata", type="Edm.String", filterable=True, facetable=True),
    ]

    semantic_config = SemanticConfiguration(
        name="default",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=None,
            keywords_fields=None,
            content_fields=[SemanticField(field_name="content")]
        )
    )

    semantic_search = SemanticSearch(configurations=[semantic_config])

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw_config",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE
                ),
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="embedding_config",
                algorithm_configuration_name="hnsw_config"
            ),
        ],
    )

    index_client = SearchIndexClient(endpoint=vector_store_address, credential=credential)
    if index_name not in index_client.list_index_names():
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
        print(f"Creating{index_name} search index")
        result = index_client.create_or_update_index(index)
        print(f"{result.name} created")
    else:
        print(f"Search index {index_name}already exists")

import glob

print("Create Search Index...")
create_search_index()
print("Processing files...")


# =================================
# 1. キーワード検索
# =================================
from langchain.retrievers import AzureCognitiveSearchRetriever

query = "源実友のお歌にはどのような特徴があったのでしょうか？"
retriever = AzureCognitiveSearchRetriever(
    service_name="20241228-azure-ai-search-study",
    index_name=index_name,
    api_key=vector_store_password,
    content_key="content",
    top_k=3,
)

docs = retriever.get_relevant_documents(query)
for doc in docs:
    print(doc.metadata["sourcepage"])
    print(doc.metadata["@search.score"])
    print(doc.page_content)

# =================================
# 2. ベクトル類似性検索
# =================================

query = "源実友のお歌にはどのような特徴があったのでしょうか？"  

docs = vector_store.similarity_search(
    query=query,
    k=3,
    search_type="similarity"
)

for doc in docs:
    print(doc.metadata["sourcepage"])
    print(doc.metadata["@search.score"])
    print(doc.page_content)

# =================================
# 2-1. 多言語対応
# =================================

query = "What were the characteristics of Minamoto Sanetomo's poetry?"  

docs = vector_store.similarity_search(
    query=query,
    k=3,
    search_type="similarity",
)

for doc in docs:
    print(doc.metadata["sourcepage"])
    print(doc.metadata["@search.score"])
    print(doc.page_content)

# =================================
# 3. ハイブリッド検索
# =================================
query = "源実友は征夷大将軍として知られているだけでなく、ある有名な趣味も持っています。それは何ですか。"  

docs = vector_store.hybrid_search(
    query=query,
    k=5
)

for doc in docs:
    print(doc.metadata["sourcepage"])
    print(doc.metadata["@search.score"])
    print(doc.page_content)

# =================================
# 3. セマンティックハイブリッド検索
# =================================
query = "１３人の合議制に含まれるメンバー一覧"  

docs = vector_store.semantic_hybrid_search(
    query=query,
    search_type="semantic_hybrid",
)

for doc in docs:
    print(doc.metadata["sourcepage"])
    print(doc.metadata["@search.score"])
    print(doc.page_content)
