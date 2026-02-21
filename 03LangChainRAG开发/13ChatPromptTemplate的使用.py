from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

# ChatPromptTemplate: 支持多种消息格式的聊天模板
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗。"),  # 系统消息
        MessagesPlaceholder("history"),              # 动态占位：插入历史消息
        ("human", "请再来一首唐诗"),                  # 用户最新消息
    ]
)

history_data = [
    ("human", "你来写一个唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗再来一个"),
    ("ai", "锄禾日当午，汗滴禾下锄，谁知盘中餐，粒粒皆辛苦"),
]

# StringPromptValue: 聊天模板invoke后返回的类型
prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()  # to_string转为字符串

model = ChatTongyi(model="qwen3-max-2026-01-23")

res = model.invoke(prompt_text)  # 输入字符串或AIMessage

print(res.content, type(res))

