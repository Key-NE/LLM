import requests
import os
os.environ['NO_PROXY'] = 'http://127.0.0.1:5030/api/llm_stream'
url = 'http://127.0.0.1:5030/api/llm_stream'
data = {
    "question": "京东健康 2023 年中期业绩总收入是多少",
    "session_id": "e10cb468-fb0e-4b30-86c7-8804b294cff0"
}
response = requests.post(url, json=data)
print(response.json())


#
# url = 'http://localhost:5000/api/llm_stream'
# # os.environ['NO_PROXY'] = 'http://localhost:5010/api/llm_invoke'
# # url = 'http://localhost:5010/api/llm_invoke'
# question = {'question': '入场方案是什么'}
#
# response = requests.post(url, json=question, stream=True)
#
# if response.status_code == 200:
#     for line in response.iter_lines():
#         if line:
#             # 去掉 'data: ' 前缀
#             data = line.decode('utf-8').replace('data: ', '')
#             print(data)
# else:
#     print(f"请求失败，状态码: {response.status_code}")
