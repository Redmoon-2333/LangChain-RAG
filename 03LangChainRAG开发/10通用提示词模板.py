"""
通用提示词模板(PromptTemplate)：动态注入变量构建个性化提示词
核心：占位符{变量名}在运行时被实际值替换，支持LCEL链式调用
"""

from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

# from_template: 通过字符串模板创建实例
# 语法：{变量名}作为占位符，运行时动态传入值
prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}, 刚生了{gender}, 你帮我起个名字，简单回答。"
)

model = Tongyi(model="qwen-max")

# 方式1：手动format（传统方式）
# prompt_text = prompt_template.format(lastname="张", gender="女儿")
# res = model.invoke(input=prompt_text)

# 方式2：LCEL链式调用（推荐）
# | 运算符实现组件串联：prompt_template输出字符串 → model输入
chain = prompt_template | model

# invoke输入字典，自动填充模板中的变量
res = chain.invoke(input={"lastname": "张", "gender": "女儿"})
print(res)
