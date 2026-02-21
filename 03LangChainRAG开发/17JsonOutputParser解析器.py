from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate

str_parser = StrOutputParser()  # AIMessage → str
json_parser = JsonOutputParser()  # AIMessage → dict/json

model = ChatTongyi(model="qwen3-max-2026-01-23")

# 第一个提示词：要求模型返回JSON格式
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

