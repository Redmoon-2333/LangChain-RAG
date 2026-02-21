"""
Chain基础使用：使用LCEL(LangChain Expression Language)构建处理链
核心：| 运算符串联多个组件，数据从前一个组件流向下一个组件
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

# 构建聊天提示词模板
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗。"),
        MessagesPlaceholder("history"),
        ("human", "请再来一首唐诗"),
    ]
)

# 历史消息数据
history_data = [
    ("human", "你来写一个唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗再来一个"),
    ("ai", "锄禾日当午，汗滴禾下锄，谁知盘中餐，粒粒皆辛苦"),
]

model = ChatTongyi(model="qwen3-max-2026-01-23")

# 组成链：要求每一个组件都是Runnable接口的子类
# LCEL: LangChain Expression Language，通过|运算符实现组件串联
chain = chat_prompt_template | model

# 方式1：invoke同步调用
# res = chain.invoke({"history": history_data})
# print(res.content)

# 方式2：stream流式输出（推荐）
# 输入字典，模板自动填充变量
for chunk in chain.stream({"history": history_data}):
    print(chunk.content, end="", flush=True)
