"""
基于 Streamlit 的知识库文件上传服务

功能：提供 Web 界面供用户上传 TXT 文件，自动解析并存储到向量知识库

核心概念：
- Streamlit：Python 的 Web 应用框架，代码重新执行机制（页面元素变化时自动重跑）
- Session State：跨页面刷新的状态存储机制

依赖安装：pip install streamlit
"""
import sys
from pathlib import Path

# 将当前脚本所在目录添加到 Python 路径，确保本地模块可导入
sys.path.insert(0, str(Path(__file__).parent))

import time
import streamlit as st
from knowledge_base import KnowledgeBaseService

# 页面标题配置
st.title("知识库更新服务")

# 文件上传组件配置
# type: 限制文件类型为 txt
# accept_multiple_files: False 表示仅接受单个文件上传
uploader_file = st.file_uploader(
    "请上传TXT文件",
    type=['txt'],
    accept_multiple_files=False,
)

# Session State 初始化
# Why: Streamlit 每次交互都会重新执行脚本，需要持久化存储服务对象
# Warning: session_state 是一个字典，首次访问不存在的键会报错，需先判断
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


if uploader_file is not None:
    # 提取文件元信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024  # 转换为 KB 便于阅读

    # 展示文件信息
    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")

    # 读取文件内容：getvalue() 返回 bytes，需解码为 utf-8 字符串
    text = uploader_file.getvalue().decode("utf-8")

    # spinner: 在代码执行期间显示加载动画，提升用户体验
    with st.spinner("载入知识库中。。。"):
        time.sleep(1)  # 模拟处理延迟，给用户感知
        result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)
        print("知识库更新完成")
