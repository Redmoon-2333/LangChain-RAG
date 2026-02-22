"""
基于 Streamlit 的智能客服问答界面

功能：提供交互式聊天界面，结合 RAG 技术实现知识增强的智能问答

核心概念：
- RAG (Retrieval-Augmented Generation): 检索增强生成，先查知识库再生成回答
- Streamlit Chat: 支持对话式交互的 UI 组件
- 流式输出: 模拟打字效果，提升用户体验
"""
import time
from rag import RagService
import streamlit as st
import config_data as config

# 页面标题和分隔线
st.title("智能客服")
st.divider()

# 初始化对话历史
# Why: 需要维护多轮对话上下文，让 AI 理解对话脉络
if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

# 初始化 RAG 服务（单例模式）
if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

# 渲染历史消息
# 遍历 session 中保存的所有消息，按角色（user/assistant）渲染
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入框（固定在页面底部）
prompt = st.chat_input()

if prompt:
    # 显示用户提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 流式响应处理
    ai_res_list = []
    with st.spinner("AI思考中..."):
        # stream(): 启用流式输出，逐字返回生成内容
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)

        # 捕获生成器输出：既用于流式展示，又用于保存完整响应
        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        # write_stream(): Streamlit 的流式输出组件，实时显示 AI 回复
        st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))
        # 将完整响应保存到 session，用于历史记录展示
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})

# join 用法示例：
# ["a", "b", "c"] -> "".join(list) -> "abc"
# ["a", "b", "c"] -> ",".join(list) -> "a,b,c"
