"""
知识库服务模块

功能：实现文档的向量化存储和管理，支持重复内容检测（MD5 去重）

核心概念：
- 文本嵌入 (Embedding): 将文本转换为高维向量，用于语义相似度计算
- 向量数据库 (Chroma): 存储和检索向量数据的高效数据库
- MD5 指纹: 用于检测重复内容，避免重复入库

架构说明：
- KnowledgeBaseService: 核心服务类，封装向量化和存储逻辑
- 辅助函数：check_md5、save_md5、get_string_md5 实现去重机制
"""
import os
import sys
from pathlib import Path
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime


def check_md5(md5_str: str) -> bool:
    """
    检查 MD5 指纹是否已存在（内容是否已处理过）

    :param md5_str: 待检查的 MD5 字符串
    :return: True 表示已处理过，False 表示未处理
    """
    if not os.path.exists(config.md5_path):
        # 文件不存在表示首次运行，肯定没有处理过
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        # 逐行比对 MD5 记录
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()  # 去除首尾空白字符
            if line == md5_str:
                return True  # 找到匹配，已处理过

        return False  # 未找到匹配


def save_md5(md5_str: str) -> None:
    """
    将 MD5 指纹记录到文件

    :param md5_str: 待记录的 MD5 字符串
    """
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8') -> str:
    """
    计算字符串的 MD5 哈希值

    Why: MD5 是一种单向哈希算法，相同内容始终产生相同哈希值
         用于快速比对内容是否相同，无需存储原始内容

    :param input_str: 输入字符串
    :param encoding: 字符编码，默认 utf-8
    :return: 32 位十六进制 MD5 字符串
    """
    # 字符串编码为字节数组
    str_bytes = input_str.encode(encoding=encoding)

    # 创建 MD5 哈希对象并更新内容
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)

    # 获取十六进制表示的哈希值
    md5_hex = md5_obj.hexdigest()

    return md5_hex


class KnowledgeBaseService(object):
    """
    知识库服务类

    职责：
    1. 管理 Chroma 向量数据库连接
    2. 提供文档上传和向量化存储功能
    3. 实现重复内容检测
    """

    def __init__(self):
        # 确保向量数据库存储目录存在
        os.makedirs(config.persist_directory, exist_ok=True)

        # 初始化 Chroma 向量数据库
        # Why: Chroma 是轻量级本地向量数据库，适合中小型项目
        self.chroma = Chroma(
            collection_name=config.collection_name,  # 集合名称（类似数据库表）
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory,  # 本地持久化路径
        )

        # 初始化递归字符文本分割器
        # Why: 长文档需要分割成小块，才能有效进行向量检索
        # chunk_overlap: 重叠区域保证分割处语义的连贯性
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,       # 每块最大字符数
            chunk_overlap=config.chunk_overlap,  # 相邻块的重叠字符数
            separators=config.separators,       # 优先使用的分割符（按优先级排序）
            length_function=len,                # 使用 Python 内置 len 计算长度
        )

    def upload_by_str(self, data: str, filename: str) -> str:
        """
        将文本内容向量化并存储到知识库

        处理流程：
        1. 计算内容 MD5，检查是否已存在
        2. 根据内容长度决定是否分割
        3. 生成元数据（来源、时间、操作人）
        4. 存入向量数据库
        5. 记录 MD5 指纹

        :param data: 文本内容
        :param filename: 来源文件名
        :return: 处理结果提示信息
        """
        # 步骤 1：MD5 去重检查
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        # 步骤 2：文本分割（长文档分割，短文档保持完整）
        if len(data) > config.max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        # 步骤 3：构建元数据（每个文本块都附加相同的元数据）
        metadata = {
            "source": filename,  # 来源文件
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 创建时间
            "operator": "Red_Moon",  # 操作人
        }

        # 步骤 4：存入向量数据库
        # add_texts: 批量添加文本，自动进行向量化
        self.chroma.add_texts(
            texts=knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],  # 每个 chunk 对应一份元数据
        )

        # 步骤 5：记录 MD5，防止重复处理
        save_md5(md5_hex)

        return "[成功]内容已经成功载入向量库"


if __name__ == '__main__':
    # 模块自测代码
    service = KnowledgeBaseService()
    r = service.upload_by_str("周杰轮222", "testfile")
    print(r)
