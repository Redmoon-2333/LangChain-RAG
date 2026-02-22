"""
向量存储服务模块
核心类：VectorStoreService，提供知识库文档加载、向量化、检索能力
基于Chroma向量数据库实现
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model.factory import embed_model
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
import os
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, csv_loader, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tools import get_abs_path


class VectorStoreService:
    """
    向量存储服务
    功能：
    1. 初始化Chroma向量数据库
    2. 加载文档（PDF/TXT/CSV）并向量化存储
    3. 提供检索接口（Retriever）
    """

    def __init__(self):
        """
        初始化向量存储：
        1. 连接Chroma数据库（指定collection和持久化目录）
        2. 配置文档分片器（RecursiveCharacterTextSplitter）
        """
        # 初始化Chroma向量数据库
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,  # 嵌入模型（通义千问）
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        # 文档分片器配置（从配置文件读取）
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],       # 每个片段的最大字符数
            chunk_overlap=chroma_conf["chunk_overlap"],  # 相邻片段的重叠字符数
            separators=chroma_conf["separators"],        # 分隔符列表
            length_function=len,                         # 计算文本长度的方式
        )

    def get_retriever(self):
        """
        获取检索器
        :return: LangChain Retriever对象，支持.invoke()方法
        """
        # search_kwargs={"k": N} 指定返回最相似的N个文档
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        加载知识库文档到向量存储
        工作流程：
        1. 遍历数据目录，获取允许的文件类型
        2. 计算每个文件的MD5，检测是否已加载（增量加载）
        3. 对新文件：读取 → 分片 → 向量化 → 存入Chroma
        4. 更新MD5缓存
        """

        def check_md5_hex(md5_for_check):
            """检查MD5是否已存在缓存文件中"""
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 首次运行：创建空文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True
            return False

        def save_md5_hex(md5_for_save):
            """将新文件的MD5追加到缓存文件"""
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_save+"\n")

        def get_file_documents(read_path: str):
            """根据文件后缀调用对应的加载器"""
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            elif read_path.endswith("pdf"):
                return pdf_loader(read_path)
            elif read_path.endswith("csv"):
                return csv_loader(read_path)
            else:
                return []

        # 获取允许的文件类型（如.txt, .pdf, .csv）
        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        # 遍历文件并处理
        for path in allowed_files_path:
            # 1. 计算MD5
            md5_hex = get_file_md5_hex(path)

            if not md5_hex:  # 处理MD5计算失败的情况
                logger.warning(f"[加载知识库] {path} MD5计算失败，跳过")
                continue

            # 2. 检查是否已加载（增量加载逻辑）
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库] {path} 内容已经存在于知识库，跳过")
                continue

            # 3. 读取并分片文档
            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库] {path} 无有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库] {path} 分片后无内容，跳过")
                    continue

                # 4. 添加到向量数据库（自动向量化）
                self.vector_store.add_documents(split_document)

                # 5. 更新MD5缓存
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库] {path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细报错堆栈，False仅记录报错str
                logger.error(f"[加载知识库] {path} 加载失败：{str(e)}", exc_info=True)
                continue


# 单元测试入口
if __name__ == '__main__':
    store = VectorStoreService()

    # 加载文档到向量库
    store.load_document()

    # 测试检索
    retriever = store.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
