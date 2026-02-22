"""
向量存储服务模块

功能：封装 Chroma 向量数据库的初始化和检索功能

核心概念：
- 向量数据库：存储文本嵌入向量，支持基于相似度的语义检索
- Retriever: LangChain 的检索器抽象，统一不同向量库的接口

设计说明：
- 将向量库操作封装为服务类，便于在 RAG 流程中复用
- 通过依赖注入传入 embedding 模型，提高灵活性
"""
from langchain_chroma import Chroma
import config_data as config


class VectorStoreService(object):
    """
    向量存储服务类

    职责：
    1. 管理 Chroma 向量数据库连接
    2. 提供检索器接口供 Chain 使用
    """

    def __init__(self, embedding):
        """
        初始化向量存储服务

        :param embedding: 嵌入模型实例，用于文本向量化
        """
        self.embedding = embedding

        # 初始化 Chroma 向量数据库
        # Why: persist_directory 指定本地持久化路径，数据可跨会话保留
        self.vector_store = Chroma(
            collection_name=config.collection_name,     # 集合名称
            embedding_function=self.embedding,          # 嵌入函数
            persist_directory=config.persist_directory,  # 持久化目录
        )

    def get_retriever(self):
        """
        获取向量检索器

        Why: as_retriever() 将向量库转换为 LangChain 标准的 Retriever 接口
             便于与 LCEL Chain 无缝集成

        :return: BaseRetriever 实例
        """
        # search_kwargs: 检索参数配置
        # k: 返回最相似的文档数量
        return self.vector_store.as_retriever(
            search_kwargs={"k": config.similarity_threshold}
        )


if __name__ == '__main__':
    # 模块自测
    from langchain_community.embeddings import DashScopeEmbeddings

    retriever = VectorStoreService(
        DashScopeEmbeddings(model=config.embedding_model_name)
    ).get_retriever()

    # 测试检索
    res = retriever.invoke("我的体重180斤，尺码推荐")
    print(res)
