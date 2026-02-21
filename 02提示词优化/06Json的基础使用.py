"""
文件功能：JSON序列化与反序列化（Python对象 <-> JSON字符串）
"""

import json

# 字典序列化
d = {"name": "周杰轮", "age": 11, "gender": "男"}
s = json.dumps(d, ensure_ascii=False)  # ensure_ascii=False: 保留中文原样
print(s)

# 列表序列化
l = [
    {"name": "周杰轮", "age": 11, "gender": "男"},
    {"name": "蔡依临", "age": 12, "gender": "女"},
    {"name": "小明", "age": 16, "gender": "男"}
]
print(json.dumps(l, ensure_ascii=False))

# 反序列化：JSON字符串 -> Python对象
json_str = '{"name": "周杰轮", "age": 11, "gender": "男"}'
json_array_str = '[{"name": "周杰轮", "age": 11, "gender": "男"}, {"name": "蔡依临", "age": 12, "gender": "女"}, {"name": "小明", "age": 16, "gender": "男"}]'

res_dict = json.loads(json_str)
print(res_dict, type(res_dict))  # dict

res_list = json.loads(json_array_str)
print(res_list, type(res_list))  # list
