import requests
import os
os.environ['NO_PROXY'] = 'http://localhost:5000/api/answer'
url = 'http://localhost:5000/api/answer'
data = {"question": "入场方案是什么"}

response = requests.post(url, json=data)
if response.status_code == 200:
    answer = response.json()["answer"]
    print(answer)
else:
    print(f"请求失败，状态码：{response.status_code}")