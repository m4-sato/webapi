# filename: main.py
import os
from openai import AzureOpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = "2024-08-01-preview"
)
deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT')


# --------------------
# ② リクエストボディ用のPydanticモデル
# --------------------
class SummarizeRequest(BaseModel):
    text: str

# prompt.txt を読み込む
with open(r"C:\Users\mssst\Git\webapi\webapi\app\prompt.txt", "r", encoding="utf-8") as f:
    LONG_PROMPT = f.read()

# --------------------
# ③ 要約を実施するエンドポイント
# --------------------
@app.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    """
    ユーザから受け取ったテキストをAzure OpenAIのChatCompletion機能を使って要約します。
    """
    # ユーザの入力テキスト
    user_input_text = request.text

    # ChatCompletion API呼び出し
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are an AI assistant that summarizes text."},
            {
                "role": "user",
                "content": LONG_PROMPT
            },
            {
                "role": "user",
                "content": user_input_text
            }
        ],
    max_tokens=800,
    temperature=0.05,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0
    )

    # レスポンスから要約結果を取得
    summary_text = response.choices[0].message.content
    return {"summary": summary_text}
