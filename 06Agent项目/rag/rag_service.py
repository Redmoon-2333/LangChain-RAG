"""
RAG服务模块：检索增强生成
核心类：RagSummarizeService，提供基于向量检索的问答能力
"""

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger
from utils.config_handler import prompts_conf
from langchain_core.runnables import RunnableLambda
from utils.chain_debug import print_prompt
from model.factory import chat_model
from utils.path_tools import get_abs_path


class RagSummarizeService:
    """
    RAG检索增强服务
    工作流程：接收用户问题 → 向量检索 → 构建上下文 → 调用LLM生成摘要
    """

    # 类变量缓存：避免多个实例重复读取提示词文件
    _PROMPT_TEXT: str = None

    def __init__(self, vector_store: VectorStoreService):
        """
        初始化RAG服务
        :param vector_store: 向量存储服务实例（提供retriever）
        """
        self.vector_store = vector_store
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = self._load_prompt_text()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _load_prompt_text(self) -> str:
        """
        从配置文件加载提示词模板（带缓存）
        Warning: 文件必须为UTF-8编码
        """
        if self._PROMPT_TEXT is not None:
            # 避免重复创建对象的重复读文件加载，从缓存读取
            return self._PROMPT_TEXT

        path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
        try:
            with open(path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
        except PermissionError:
            logger.error(f"无权限读取提示词文件：{path}")
            raise PermissionError(f"无权限读取提示词文件：{path}")
        except UnicodeDecodeError:
            logger.error(f"提示词文件编码错误（需UTF-8）：{path}")
            raise ValueError(f"提示词文件编码错误（需UTF-8）：{path}")
        except Exception as e:
            logger.error(f"读取提示词文件失败：{str(e)}")
            raise RuntimeError(f"读取提示词文件失败：{str(e)}")

        if not prompt_text:
            logger.error(f"提示词文件内容为空：{path}")
            raise ValueError(f"提示词文件内容为空：{path}")

        # 记录缓存
        self._PROMPT_TEXT = prompt_text
        return prompt_text

    def _init_chain(self):
        """
        初始化LCEL Chain：PromptTemplate → ChatModel → StrOutputParser
        """
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retrieve_docs(self, query: str) -> list[Document]:
        """
        检索相关文档
        :param query: 用户查询字符串
        :return: 文档列表（按相似度排序）
        """
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """
        RAG核心方法：检索+生成
        :param query: 用户查询
        :return: 生成的回答字符串
        """
        # 构建输入字典
        # key: input, for user query
        # key: context, 参考资料
        input_dict = {}

        # 1. 向量检索：从知识库获取相关文档
        context_docs = self.retrieve_docs(query)

        # 2. 构建上下文：将检索到的文档拼接为字符串
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        # 3. 构建输入：用户问题 + 检索到的上下文
        input_dict["input"] = query
        input_dict["context"] = context

        # 4. 调用Chain生成回答
        return self.chain.invoke(input_dict)


# 单元测试入口
if __name__ == '__main__':
    vs = VectorStoreService()
    rag = RagSummarizeService(vs)

    print(rag.rag_summarize("小户型适合哪种扫地机器人？"))
