import os, json
from typing import Sequence

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory


class FileChatMessageHistory(BaseChatMessageHistory):
    """文件持久化的会话记忆：将聊天历史存储到本地JSON文件
    
    与InMemoryChatMessageHistory对比：内存存储重启丢失，文件存储持久化保留
    """
    
    def __init__(self, session_id, storage_path):
        self.session_id = session_id        # 会话ID，作为文件名
        self.storage_path = storage_path    # 存储目录路径
        self.file_path = os.path.join(self.storage_path, self.session_id)  # 完整文件路径

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)  # 确保目录存在

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """追加消息并持久化到文件
        
        Sequence: 序列类型，接受list/tuple等可迭代对象
        """
        all_messages = list(self.messages)  # 读取已有历史
        all_messages.extend(messages)       # 追加新消息

        # 持久化：BaseMessage → dict → JSON文件
        new_messages = [message_to_dict(message) for message in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)

    @property  # 将方法转为属性，调用时无需括号：obj.messages 而非 obj.messages()
    def messages(self) -> list[BaseMessage]:
        """读取历史消息：JSON文件 → dict → BaseMessage列表"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)  # list[dict]
                return messages_from_dict(messages_data)  # dict → BaseMessage
        except FileNotFoundError:
            return []  # 首次会话无历史文件

    def clear(self) -> None:
        """清空会话历史：写入空列表"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)



model = ChatTongyi(model="qwen3-max-2026-01-23")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你需要根据会话历史回应用户问题。对话历史："),
        MessagesPlaceholder("chat_history"),  # 动态注入历史消息
        ("human", "请回答如下问题：{input}")   # 用户当前输入
    ]
)

str_parser = StrOutputParser()


def print_prompt(full_prompt):
    """调试函数：打印最终生成的prompt"""
    print("="*20, full_prompt.to_string(), "="*20)
    return full_prompt


base_chain = prompt | print_prompt | model | str_parser


def get_history(session_id):
    """会话历史工厂函数：根据session_id创建FileChatMessageHistory实例"""
    return FileChatMessageHistory(session_id, "./chat_history")


# RunnableWithMessageHistory: 为基础链添加会话历史自动管理功能
# 自动流程：invoke前读取历史 → 注入prompt → invoke后保存新消息
conversation_chain = RunnableWithMessageHistory(
    base_chain,                         # 被增强的基础链
    get_history,                        # 会话历史工厂函数
    input_messages_key="input",         # 用户输入变量名
    history_messages_key="chat_history" # 历史消息变量名
)


if __name__ == '__main__':
    # session_config: 会话配置，session_id区分不同用户/会话
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

