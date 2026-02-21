# langchain_community: LangChain社区提供的第三方集成包
from langchain_community.llms.tongyi import Tongyi

model = Tongyi(model="qwen-max")  # model参数指定使用的通义模型

res = model.invoke(input="你是谁呀能做什么？")  # invoke: 同步调用，返回完整响应

print(res)
