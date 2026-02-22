"""
项目全局配置文件

功能：集中管理所有配置参数，便于维护和修改

设计原则：
- 路径使用 Path 对象确保跨平台兼容性（Windows/Linux/Mac）
- 敏感信息（API Key）应通过环境变量传入，本文件仅保留非敏感配置
"""
from pathlib import Path

# 获取当前文件所在目录，作为项目根路径
_BASE_DIR = Path(__file__).parent

# MD5 记录文件路径：用于去重，记录已处理的文件指纹
md5_path = str(_BASE_DIR / "md5.text")


# Chroma 向量数据库配置
collection_name = "rag"  # 数据库集合（表）名称
persist_directory = str(_BASE_DIR / "chroma_db")  # 本地持久化存储路径


# 文本分割器配置（RecursiveCharacterTextSplitter）
chunk_size = 1000        # 每个文本块的最大字符数
chunk_overlap = 100      # 相邻文本块的重叠字符数（保证语义连贯性）
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]  # 分割优先级
def max_split_char_number():
    """文本分割的阈值：超过此长度才进行分割"""
    return 1000


# 向量检索配置
similarity_threshold = 1  # 返回最相似的文档数量（top-k）

# 模型配置（阿里云 DashScope / 通义千问）
embedding_model_name = "text-embedding-v4"      # 文本嵌入模型
chat_model_name = "qwen3-max-2026-01-23"        # 对话生成模型

# LangChain Session 配置
session_config = {
    "configurable": {
        "session_id": "user_001",  # 会话标识，用于区分不同用户的对话历史
    }
}
