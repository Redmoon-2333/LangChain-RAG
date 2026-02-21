"""
文件功能：演示OpenAI SDK基础使用（初始化 + 非流式对话 + 带历史消息）
"""

from openai import OpenAI

# 默认从环境变量OPENAI_API_KEY读取
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云百炼兼容模式
)

# messages: 按时间顺序排列的对话列表
# 角色说明：system(系统设定) / assistant(历史回复) / user(用户提问)
response = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[
        {"role": "system", "content": "你是一个Python编程专家"},  # 设定助手角色
        {"role": "assistant", "content": "好的我是编程专家，你想要问什么"},  # 历史对话
        {"role": "user", "content": "输出a+b的代码"}  # 当前问题
    ]
)

print(response.choices[0].message.content)  # 非流式：一次性返回完整响应
