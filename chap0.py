# ライブラリのインストールとバージョン確認
import azure.search.documents
print("azure.search.documents", azure.search.documents.__version__)
import azure.ai.formrecognizer
print("azure.ai.formrecognizer", azure.ai.formrecognizer.__VERSION__)
import azure.storage.blob
print("azure.storage.blob", azure.storage.blob.__version__)
import os
import io
import time
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

import openai
from dotenv import load_dotenv


load_dotenv()

# =================================
# 1. 接続設定
# =================================

# Blob Storage
azure_storage_container = "contain"
azure_blob_connection = ""

# Azure AI Search
search_service_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
search_service_admin_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
index_name = "gptkbindex"
search_analyzer_name = "ja.lucene"
credential = AzureKeyCredential(search_service_admin_key)

# Azure AI Document Intelligence
document_intelligence_endpoint = os.getenv("AZURE_AI_DOCUMENT_ENDPOINT")
document_intelligence_key = os.getenv("AZURE_AI_DOCUMENT_API_KEY")
document_intelligence_creds = AzureKeyCredential(document_intelligence_key)

# # AzureOpenAI
# client = AzureOpenAI(
#     model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
#     api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
#     api_version="2023-05-15",
#     azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")
#     # chunk_size=2048
#     )


# 1) Azure Resource に紐づく情報を設定
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")  # 例: https://<your-resource-name>.openai.azure.com/
openai.api_version = "2023-05-15"  # (エンドポイント作成時に指定したバージョン)
openai.api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY")

# 2) Embeddings を呼び出す
response = openai.Embedding.create(
    input="Your text string goes here",
    engine="text-embedding-ada-002"  # デプロイ時の「モデル名」ではなく「デプロイ名」を書く
)

print(response)

# response = client.embeddings.create(
#     input = "Your text string goes here",
#     model= "text-embedding-3-small"
# )

# print(response.model_dump_json(indent=2))
# ====================================
# 2. 検索インデックスの定義
# ====================================

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

    index_client = SearchIndexClient(endpoint=search_service_endpoint, credential=credential)
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


def remove_from_index(filename):
    print(f"Removing sections from '{filename or '<all>'}' from search index '{index_name}'")
    search_client = SearchClient(endpoint=search_service_endpoint,
                                index_name=index_name,
                                credential=credential)
    while True:
        filter = None if filename is None else f"sourcefile eq '{os.path.basename(filename)}'"
        r = search_client.search("", filter=filter, top=1000, include_total_count=True)
        if r.get_count==0:
            break
        r = search_client.delete_documents(documents=[{"id": d["id"]} for d in r])
        print(f"\tRemoved{len(r)} sections from index")
        time.sleep(2)

# =============================================
# 3. Azure Blob StorageにPDFファイルをアップロード
# =============================================

from azure.storage.blob import BlobServiceClient
from pypdf import PdfReader, PdfWriter

def blob_name_from_file_page(filename, page = 0):
    if os.path.splitext(filename)[1].lower() == ".pdf":
        return os.path.splitext(os.path.basename(filename))[0] + f"-{page}" + ".pdf"
    else:
        return os.path.basename(filename)

def upload_blobs(filename):
    blob_service_client = BlobServiceClient.from_connection_string(azure_blob_connection_string)
    blob_container = blob_service_client.get_container_client(azure_storage_container)
    if not blob_container.exists():
        blob_container.create_container()
    
    if os.path.splitext(filename)[1].lower() == ".pdf":
        reader = PdfReader(filename)
        pages = reader.pages
        for i in range(len(pages)):
            blob_name = blob_name_from_file_page(filename, i)

            f = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(pages[i])
            writer.write(f)
            f.seek(0)
            blob_container.upload_blob(blob_name, f, overwrite=True)
    
    else:
        blob_name = blob_name_from_file_page(filename)
        with open(filename, "rb") as data:
            blob_container.upload_blob(blob_name, data, overwrite=True)

# ===========================================
# 4. Azure Document Intelligenceを利用したOCR
# ===========================================

from azure.ai.formrecognizer import DocumentAnalysisClient
import html
import jsonpickle

def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html

def get_document_text(filename):
    offset = 0
    page_map = []

    print(f"Extracting text from '{filename}' using Azure AI Document Intelligence")
    form_recognizer_client = DocumentAnalysisClient(endpoint=document_intelligence_endpoint, credential=document_intelligence_creds, headers={"x-ms-useragent":"azure-search-chat-demo/1.0.0"})
    with open(filename, "rb") as f:
        poller = form_recognizer_client.begin_analyze_document("prebuilt-layout", document = f)
    form_recognizer_results = poller.result()

    for page_num, page in enumerate(form_recognizer_results.pages):
        tables_on_page = [table for table in form_recognizer_results.tables if table.bounding_regions[0].page_number == page_num + 1]

        page_offset = page.spans[0].offset
        page_length = page.spans[0].length
        table_chars = [-1]*page_length
        for span in table.spans:
            for i in range(span.length):
                idx = span.offset - page_offset + i
                if idx >=0 and idx < page_length:
                    table_chars[idx] = table_id
        
        page_text = ""
        added_tables = set()
        for idx, table_id in enumerate(table_chars):
            if table_id == -1:
                page_text += form_recognizer_results.content[page_offset + idx]
            elif table_id not in added_tables:
                page_text += table_to_html(tables_on_page[table_id])
                added_tables.add(table_id)
        
        page_text = " "
        page_map.append((page_num, offset, page_text))
        offset += len(page_text)
    
    return page_map


# ===========================================
# 5. Embeddings生成関数の定義
# ===========================================

from tenacity import retry, stop_after_attempt, wait_random_exponential

# client = AzureOpenAI(
#     api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),  
#     api_version = "2023-05-15",
#     azure_endpoint =os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT") 
# )

def before_retry_sleep(retry_state):
    print("Rate limited on the OpenAI embedding API, sleeping before retrying...")

@retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(15), before_sleep=before_retry_sleep)
def compute_embedding(text):
    return openai.Embedding.create(input = [text], model=model).data[0].embedding


# ===========================================
# 6. チャンク分割
# ===========================================

MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100

def split_text(page_map, filename):
    SENTENCE_EMBEDDINGS = [".", "[", "?"]
    WORDS_BREAKS = [".", ";", ":", "", "(", ")", "[", "]", "{", "}", "\t", "\n"]
    print(f"Splitting '{filename}' into sections")

    def find_page(offset):
        num_pages = len(page_map)
        for i in range(num_pages - 1):
            if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                return i
        return num_pages - 1
    
    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_EMBEDDINGS and last_word > 0:
                end = last_word
        if end < length:
            end += 1
        
        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH -2 * SENTENCE_SEARCH_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_EMBEDDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1
        

        section_text = all_text[start:end]
        yield (section_text, find_page(start))

        last_table_start = section_text.rfind("<table>")
        if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table>")):
            print(f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}")
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP
    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))

# ==============================================
# 6. インデックスに登録するドキュメントを作成    
# ==============================================
import re
import base64
import json

def filename_to_id(filename):
    filename_ascii = re.sub("[^0-9a-zA-Z_-]", "_", filename)
    filename_hash = base64.b16encode(filename.encode('utf-8')).decode('ascii')
    return f"file-{filename_ascii}-{filename_hash}"

def create_sections(filename, page_map, use_vectors, category):
    file_id = filename_to_id(filename)
    for i, (content, pagenum) in enumerate(split_text(page_map, filename)):
        section = {
            "id": f"{file_id}-page-{i}",
            "content": content,
            "category": category,
            "sourcepage": blob_name_from_file_page(filename, pagenum),
            "sourcefile": filename,
            "metadata": json.dumps({"page": pagenum, "sourcepage": blob_name_from_file_page(filename, pagenum)})
        }

        section["embedding"] = compute_embedding(content)
        yield section

# ==============================================
# 7. チャンクをインデックス化  
# ==============================================

def index_sections(filename, sections):
    search_client = SearchClient(
        endpoint = search_service_endpoint, index_name=index_name, credential=credential
    )
    i = 0
    batch = []
    for s in sections:
        batch.append(s)
        i += 1
        if i % 1000 == 0:
            results = search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
            batch = []
    
    if len(batch) > 0:
        results = search_client.upload_documents(documents=batch)
        succeeded = sum([1 for r in results if r.succeeded])
        print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")


# ==============================================
# 8. メイン処理  
# ==============================================

import glob

print("Create Search Index...")
create_search_index()
print("Processing files...")

path_pattern = "../data/*.pdf"
for filename in glob.glob(path_pattern):
    print(f"Processing'{filename}'")
    try:
        upload_blobs(filename)
        remove_from_index(filename)
        page_map = get_document_text(filename)
        category = os.path.basename(os.path.dirname(filename))
        sections = create_sections(
            os.path.basename(filename), page_map, False, category
        )
        index_sections(os.path.basename(filename), sections)
    
    except Exception as e:
        print(f"\tGot an error while reading {filename} -> {e} --> skipping file")
