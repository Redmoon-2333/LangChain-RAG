from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.tongyi import Tongyi
from langchain_community.chat_models.tongyi import ChatTongyi

"""
LangChain组件继承链：
PromptTemplate -> StringPromptTemplate -> BasePromptTemplate -> RunnableSerializable -> Runnable
FewShotPromptTemplate -> StringPromptTemplate -> BasePromptTemplate  -> RunnableSerializable -> Runnable
ChatPromptTemplate -> BaseChatPromptTemplate -> BasePromptTemplate  -> RunnableSerializable -> Runnable
Tongyi -> BaseLLM -> BaseLanguageModel -> RunnableSerializable -> Runnable
ChatTongyi -> BaseChatModel -> BaseLanguageModel -> RunnableSerializable -> Runnable

核心：所有组件都实现Runnable接口，支持|串联、invoke/stream/batch方法
"""


template = PromptTemplate.from_template("我的邻居是：{lastname}，最喜欢：{hobby}")

res = template.format(lastname="张大明", hobby="钓鱼")  # format: 直接返回填充后的字符串
print(res, type(res))


res2 = template.invoke({"lastname": "周杰轮", "hobby": "唱歌"})  # invoke: 返回PromptValue对象（可转换为字符串）
print(res2, type(res2))
