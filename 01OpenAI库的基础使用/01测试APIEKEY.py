"""
文件功能：测试API密钥配置 + 演示流式输出基础用法
"""

import os
from openai import OpenAI

# 环境变量读取API密钥，避免硬编码泄露
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云百炼兼容模式
)

# 流式输出：逐块接收响应，提升用户体验
completion = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    stream=True
)

for chunk in completion:
    print(chunk.choices[0].delta.content, end="", flush=True)  # end=""不换行，flush实时显示
