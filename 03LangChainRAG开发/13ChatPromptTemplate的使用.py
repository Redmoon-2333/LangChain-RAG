"""
ChatPromptTemplate聊天提示词模板：支持多种消息格式的模板
与PromptTemplate对比：专用于聊天场景，支持system/human/ai消息角色区分
核心：MessagesPlaceholder实现动态历史消息注入
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

# ChatPromptTemplate支持多种消息格式
# from_messages: 接收消息列表，支持元组简写和MessagePlaceholder
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗。"),  # 系统消息：设定AI角色
        MessagesPlaceholder("history"),              # 动态占位符：插入历史消息
        ("human", "请再来一首唐诗"),                  # 用户最新消息
    ]
)

# 历史消息数据：元组列表
history_data = [
    ("human", "你来写一个唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗再来一个"),
    ("ai", "锄禾日当午，汗滴禾下锄，谁知盘中餐，粒粒皆辛苦"),
]

# invoke返回StringPromptValue，to_string()转换为字符串供模型使用
prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()

model = ChatTongyi(model="qwen3-max-2026-01-23")

# 输入字符串或AIMessage皆可
res = model.invoke(prompt_text)

print(res.content, type(res))

