"""
配置处理器模块
统一加载项目中的YAML配置文件（rag.yml、chroma.yml、prompts.yml、agent.yml）
"""

import yaml
from utils.path_tools import get_abs_path


class ConfigHandler(object):
    """配置处理器：静态方法类，提供各类配置文件的加载方法"""

    @staticmethod
    def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding="utf-8"):
        """加载RAG配置：模型名称、嵌入模型名称等"""
        with open(config_path, "r", encoding=encoding) as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    @staticmethod
    def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding="utf-8"):
        """加载Chroma配置：向量数据库参数、分片参数等"""
        with open(config_path, "r", encoding=encoding) as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    @staticmethod
    def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding="utf-8"):
        """加载提示词配置：提示词文件路径等"""
        with open(config_path, "r", encoding=encoding) as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    @staticmethod
    def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding="utf-8"):
        """加载Agent配置：外部数据路径等"""
        with open(config_path, "r", encoding=encoding) as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)


# 模块级加载：项目启动时自动加载所有配置
rag_conf = ConfigHandler.load_rag_config()
chroma_conf = ConfigHandler.load_chroma_config()
prompts_conf = ConfigHandler.load_prompts_config()
agent_conf = ConfigHandler.load_agent_config()
