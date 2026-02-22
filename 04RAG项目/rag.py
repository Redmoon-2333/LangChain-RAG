"""
RAG (Retrieval-Augmented Generation) 服务模块

功能：实现检索增强生成流程，结合向量检索和大模型生成能力

核心概念：
- RAG: 先检索相关知识，再基于知识生成回答，解决大模型幻觉问题
- LCEL (LangChain Expression Language): 使用 | 管道符组合链式调用
- Runnable: LangChain 的可运行组件抽象，支持流式、批处理、异步等

Chain 流程图：
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   用户输入   │ -> │  同时执行   │ -> │  格式化输出  │
│  {input}    │    │  1. 直通    │    │  合并结果   │
└─────────────┘    │  2. 检索    │    └──────┬──────┘
                   └─────────────┘           │
                         │                   │
                         v                   v
                   ┌─────────────┐    ┌─────────────┐
                   │  向量检索    │    │  Prompt模板 │
                   │  Retriever  │ -> │  组装消息   │
                   └─────────────┘    └──────┬──────┘
                                             │
                                             v
                                       ┌─────────────┐
                                       │  大模型生成  │
                                       │ ChatTongyi  │
                                       └──────┬──────┘
                                              │
                                              v
                                       ┌─────────────┐
                                       │  字符串解析  │
                                       │StrOutputParser│
                                       └─────────────┘
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi


def print_prompt(prompt):
    """
    调试辅助函数：打印最终发送给模型的 Prompt

    Why: 便于排查 Prompt 组装是否符合预期，是调试 RAG 的关键工具

    :param prompt: ChatPromptValue 对象
    :return: 原样返回 prompt，不影响链式调用
    """
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)

    return prompt


class RagService(object):
    """
    RAG 服务类

    职责：组装检索-生成链路，管理向量检索、Prompt 模板、大模型调用
    """

    def __init__(self):
        # 初始化向量检索服务
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        # 定义 Prompt 模板
        # Why: 使用 from_messages 支持多角色消息（system/user/assistant）
        # MessagesPlaceholder: 动态插入对话历史的位置标记
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，"
                 "简洁和专业的回答用户问题。参考资料:{context}。"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),  # 历史消息占位符
                ("user", "请回答用户提问：{input}")
            ]
        )

        # 初始化大模型（通义千问）
        self.chat_model = ChatTongyi(model=config.chat_model_name)

        # 组装完整 Chain
        self.chain = self.__get_chain()

    def __get_chain(self):
        """
        构建 RAG 执行链

        Chain 组装逻辑：
        1. 并行处理：用户输入同时传递给 context（检索）和 input（直通）
        2. 检索处理：输入 -> Retriever -> 格式化文档
        3. Prompt 组装：合并 input、context、history
        4. 模型调用：Prompt -> ChatModel -> 解析输出

        :return: 可运行的 Chain 对象
        """
        # 获取向量检索器
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]) -> str:
            """
            将检索到的文档格式化为字符串

            :param docs: Document 对象列表
            :return: 格式化后的参考文本
            """
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value: dict) -> str:
            """
            从输入字典中提取检索查询文本

            :param value: 包含 input 的字典
            :return: 检索查询字符串
            """
            return value["input"]

        def format_for_prompt_template(value: dict) -> dict:
            """
            格式化数据以匹配 Prompt 模板

            Why: Prompt 模板需要 {input, context, history} 三个变量
                 但上游输出结构不同，需要转换

            :param value: 上游输出的字典
            :return: 符合 Prompt 模板输入格式的字典
            """
            new_value = {}
            new_value["input"] = value["input"]["input"]  # 嵌套结构解包
            new_value["context"] = value["context"]       # 检索结果
            new_value["history"] = value["input"]["history"]  # 对话历史
            return new_value

        # 核心 Chain 组装（LCEL 语法）
        # RunnablePassthrough: 直通，不做处理原样传递
        # RunnableLambda: 包装自定义函数为 Runnable
        # | 管道符：将左侧输出作为右侧输入
        chain = (
            {
                "input": RunnablePassthrough(),  # 直通用户输入
                "context": RunnableLambda(format_for_retriever) | retriever | format_document  # 检索分支
            }
            | RunnableLambda(format_for_prompt_template)  # 格式转换
            | self.prompt_template   # 组装 Prompt
            | print_prompt           # 调试打印（可选）
            | self.chat_model        # 调用大模型
            | StrOutputParser()      # 解析为字符串
        )

        # 包装为带历史记录的对话链
        # Why: RunnableWithMessageHistory 自动管理对话历史的存取
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,              # 历史记录获取函数
            input_messages_key="input",    # 输入消息的键名
            history_messages_key="history",  # 历史消息的键名
        )

        return conversation_chain


if __name__ == '__main__':
    # 模块自测
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }

    res = RagService().chain.invoke({"input": "针织毛衣如何保养？"}, session_config)
    print(res)
