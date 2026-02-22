"""
模型工厂模块
提供ChatModel和Embeddings模型的统一创建入口
采用工厂模式，支持多模型切换
"""

from abc import ABC
from abc import abstractmethod
from langchain_community.chat_models.tongyi import ChatTongyi, BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from typing import Optional
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    """模型工厂抽象基类"""

    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """生成模型实例的抽象方法"""
        pass


class ChatModelFactory(BaseModelFactory):
    """聊天模型工厂：生成通义千问ChatModel"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # 从配置文件读取模型名称
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    """嵌入模型工厂：生成通义千问Embeddings"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # 从配置文件读取嵌入模型名称
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# 模块级单例：项目启动时直接创建模型实例
# 用途：供其他模块导入使用（如rag_service、react_agent）
chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
