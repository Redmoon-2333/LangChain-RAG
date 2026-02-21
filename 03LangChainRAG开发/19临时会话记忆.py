"""
临时会话记忆：使用InMemoryChatMessageHistory实现内存存储的会话历史
特点：会话数据存储在内存中，进程结束后数据丢失，适用于开发测试
核心：RunnableWithMessageHistory自动管理历史消息的读取和保存
"""

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

model = ChatTongyi(model="qwen3-max-2026-01-23")

# ChatPromptTemplate支持消息占位符，用于注入历史消息
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你需要根据会话历史回应用户问题。对话历史："),
        MessagesPlaceholder("chat_history"),  # 占位符：自动填充历史消息
        ("human", "请回答如下问题：{input}")   # 占位符：用户当前输入
    ]
)

str_parser = StrOutputParser()

# 打印中间产物：查看prompt最终形态
def print_prompt(full_prompt):
    print("="*20, full_prompt.to_string(), "="*20)
    return full_prompt


base_chain = prompt | print_prompt | model | str_parser  # 基础链


store = {}  # 内存存储：key=session_id，value=InMemoryChatMessageHistory

# 会话历史工厂函数：根据session_id获取或创建历史记录
def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()  # 首次创建空历史
    return store[session_id]

# RunnableWithMessageHistory: 为基础链添加会话历史功能
# Warning: 内存存储仅适合开发测试，生产环境应使用数据库持久化
conversation_chain = RunnableWithMessageHistory(
    base_chain,                     # 被增强的原有chain
    get_history,                    # 会话历史工厂函数
    input_messages_key="input",      # 用户输入在模板中的变量名
    history_messages_key="chat_history"  # 历史消息在模板中的变量名
)


if __name__ == '__main__':
    # session_config: LangChain会话配置，session_id标识不同用户/会话
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }

    res = conversation_chain.invoke({"input": "小明有2个猫"}, session_config)
    print("第1次执行：", res)

    res = conversation_chain.invoke({"input": "小刚有1只狗"}, session_config)
    print("第2次执行：", res)

    res = conversation_chain.invoke({"input": "总共有几个宠物"}, session_config)
    print("第3次执行：", res)
