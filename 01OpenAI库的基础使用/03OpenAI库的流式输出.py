"""
文件功能：演示流式输出（与02对比：stream=True + 迭代处理）
"""

from openai import OpenAI

# 【相同部分】与02完全一致
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 【差异】stream=True 开启流式输出
response = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[
        {"role": "system", "content": "你是一个Python编程专家"},
        {"role": "assistant", "content": "好的我是编程专家，你想要问什么"},
        {"role": "user", "content": "输出a+b的代码"}
    ],
    stream=True  # 关键差异：开启流式输出
)

# 【差异】for循环逐块迭代处理
for chunk in response:
    print(
        chunk.choices[0].delta.content,  # delta.content: 增量文本片段
        end=" ",  # 空格分隔
        flush=True  # 实时显示
    )