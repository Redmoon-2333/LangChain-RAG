"""
文件功能：演示多轮对话 + 流式输出（与03对比：messages包含更长历史）
"""

from openai import OpenAI

# 【相同部分】与之前文件一致
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 【差异】messages包含完整多轮对话历史，用于上下文理解
response = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[
        {"role": "system", "content": "你是AI助理，回答很简洁"},
        {"role": "user", "content": "小明有2条宠物狗"},  # 第一轮用户
        {"role": "assistant", "content": "好的"},  # 第一轮助手
        {"role": "user", "content": "小红有3只宠物猫"},  # 第二轮用户
        {"role": "assistant", "content": "好的"},  # 第二轮助手
        {"role": "user", "content": "总共有几个宠物吖"},  # 需要结合前两轮信息回答
    ],
    stream=True  # 【相同部分】开启流式输出
)

# 【相同部分】流式处理与03一致
for chunk in response:
    print(
        chunk.choices[0].delta.content,
        end=" ",
        flush=True
    )