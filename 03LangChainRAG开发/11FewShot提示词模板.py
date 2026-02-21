"""
文件功能：FewShotPromptTemplate少样本模板（与10对比：包含示例）
"""

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.llms.tongyi import Tongyi

# 示例模板：定义示例的输出格式
example_template = PromptTemplate.from_template("单词：{word}, 反义词：{antonym}")

# 示例数据：提供少样本学习示例
examples_data = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"},
]

# FewShotPromptTemplate: prefix(指令) + examples(示例) + suffix(用户输入)
few_shot_template = FewShotPromptTemplate(
    example_prompt=example_template,  # 示例格式模板
    examples=examples_data,            # 示例数据集
    prefix="告知我单词的反义词，我提供如下的示例：",  # 前缀指令
    suffix="基于前面的示例告知我，{input_word}的反义词是？",  # 后缀用户输入
    input_variables=['input_word']     # 用户输入变量列表
)

prompt_text = few_shot_template.invoke(input={"input_word": "左"}).to_string()
print(prompt_text)

model = Tongyi(model="qwen-max")
print(model.invoke(input=prompt_text))
