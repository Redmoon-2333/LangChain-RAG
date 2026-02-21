"""
JsonOutputParser解析器：将模型输出从AIMessage转换为Python字典/JSON对象
与StrOutputParser对比：解析结构化数据，适用于API返回、数据提取等场景
核心：依赖Pydantic模型定义输出格式，确保输出符合预期结构
"""

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate

# StrOutputParser: AIMessage → str
str_parser = StrOutputParser()
# JsonOutputParser: AIMessage → dict/json
json_parser = JsonOutputParser()

model = ChatTongyi(model="qwen3-max-2026-01-23")

# 第一个提示词：要求模型返回JSON格式
# 关键技巧：在提示词中明确格式要求，提高模型输出准确性
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请帮忙起名字，"
    "并封装为JSON格式返回给我。要求key是name，value就是你起的名字，请严格遵守格式要求。"
)

# 第二个提示词：解析名字含义
second_prompt = PromptTemplate.from_template(
    "姓名：{name}，请帮我解析含义。"
)

# 构建链：first_prompt → model → json_parser → second_prompt → model → str_parser
# json_parser输出dict，key作为变量传递给second_prompt
chain = first_prompt | model | json_parser | second_prompt | model | str_parser

for chunk in chain.stream({"lastname": "张", "gender": "女儿"}):
    print(chunk, end="", flush=True)

