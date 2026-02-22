"""
文件持久化的对话历史存储模块

功能：实现基于本地文件的对话历史存储，支持多会话管理

核心概念：
- BaseChatMessageHistory: LangChain 的对话历史抽象基类
- 自定义存储后端：将对话记录保存到本地 JSON 文件，而非内存

适用场景：
- 需要长期保存对话记录
- 应用重启后仍需恢复对话上下文
"""
import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


def get_history(session_id):
    """
    工厂函数：根据 session_id 创建对应的历史记录存储实例

    :param session_id: 会话唯一标识
    :return: FileChatMessageHistory 实例
    """
    return FileChatMessageHistory(session_id, "./chat_history")


class FileChatMessageHistory(BaseChatMessageHistory):
    """
    基于文件的对话历史存储实现

    继承自 LangChain 的 BaseChatMessageHistory，实现自定义持久化逻辑
    """

    def __init__(self, session_id, storage_path):
        """
        初始化文件存储

        :param session_id: 会话唯一标识，作为文件名
        :param storage_path: 存储文件夹路径
        """
        self.session_id = session_id
        self.storage_path = storage_path
        # 完整文件路径：storage_path/session_id
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # 确保存储目录存在（exist_ok=True 避免重复创建报错）
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        添加消息到历史记录

        Why: Sequence 是抽象基类，比 list 更灵活，可接受 list、tuple 等

        :param messages: 待添加的消息序列
        """
        # 合并已有消息和新消息
        all_messages = list(self.messages)  # 读取现有记录
        all_messages.extend(messages)       # 追加新消息

        # 序列化：BaseMessage 对象 -> 字典 -> JSON 字符串
        # Why: LangChain 提供 message_to_dict 工具函数，确保序列化兼容性
        new_messages = [message_to_dict(message) for message in all_messages]

        # 写入文件（覆盖写入，保存完整历史）
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f, ensure_ascii=False, indent=2)

    @property
    def messages(self) -> list[BaseMessage]:
        """
        获取当前会话的所有历史消息

        @property 装饰器：将方法转换为属性访问方式（无需括号调用）

        :return: BaseMessage 对象列表
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)  # JSON 字符串 -> Python 字典列表
                # 反序列化：字典 -> BaseMessage 对象
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            # 文件不存在表示新会话，返回空列表
            return []

    def clear(self) -> None:
        """清空当前会话的历史记录"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
