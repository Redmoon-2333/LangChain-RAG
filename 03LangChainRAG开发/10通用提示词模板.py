"""
文件功能：PromptTemplate通用模板（动态注入变量）
"""

from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

# from_template: 模板占位符 {变量名}，创建模板实例
prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}, 刚生了{gender}, 你帮我起个名字，简单回答。"
)

model = Tongyi(model="qwen-max")

# 方式1：手动format
# prompt_text = prompt_template.format(lastname="张", gender="女儿")
# res = model.invoke(input=prompt_text)

# 方式2：LCEL链式调用（推荐），| 运算符实现组件串联
chain = prompt_template | model  # prompt_template输出字符串 → model输入

res = chain.invoke(input={"lastname": "张", "gender": "女儿"})  # 输入字典自动填充模板变量
print(res)
