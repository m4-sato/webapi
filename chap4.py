import os
import openai
import langchain
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools import AIPluginTool
from langchain.agents.mrkl import prompt

from requests.exceptions import ConnectionError

from dotenv import load_dotenv

load_dotenv()


# =================================
# 1.LLMの定義
# =================================
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.0
    )

# =================================
# 2.Toolの定義
# =================================
tools = load_tools(["requests_all"])
plugin_urls = [
    "http://localhost:5005/.well-known/ai-plugin.json",
    "http://localhost:5006/.well-known/ai-plugin.json",
]
tools += [AIPluginTool.from_plugin_url(url) for url in plugin_urls]

# =================================
# 3.Agentの定義
# =================================
SUFFIX = """
'Answer should be in Japanese. Use http instead of https for endpoint.
If there is no year in the reservation, use the year 2023. 
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

# =================================
# 4.Agentの実行
# =================================
try:
    result = agent_chain.run("源範頼に関連するカフェ名を検索して")
    print(result)
except ConnectionError as e:
    print("すみません、わかりません。(ConnectionError)", e)
except Exception as e:
    print("すみません、わかりません。(Error)", e)

try:
    result = agent_chain.run("源範頼に関連するカフェ名を検索して、7/1の18時に予約に空きがあるか教えて。もし空いていたら予約しておいて。")
    print(result)
except ConnectionError as e:
    print("すみません、わかりません。(ConnectionError)", e)
except Exception as e:
    print("すみません、わかりません。(Error)", e)

try:
    result = agent_chain.run("カフェかば殿に7/1の18時に予約を取って")
    print(result)
except ConnectionError as e:
    print("すみません、わかりません。(ConnectionError)", e)
except Exception as e:
    print("すみません、わかりません。(Error)", e)