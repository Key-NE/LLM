import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
device = torch.device("cuda:0")
# 加载分词器和模型
tokenizer_chat = AutoTokenizer.from_pretrained(r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B")
model_chat = AutoModelForCausalLM.from_pretrained(r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B")


# # 输入文本
# input_text = "你好"
#
# # 使用分词器对输入文本进行分词
# input_ids = tokenizer_chat(input_text, return_tensors="pt").input_ids
# # 将输入传递给模型进行推理
# output = model_chat.generate(input_ids)
#
# # 对模型的输出进行解码
# output_text = tokenizer_chat.decode(output[0], skip_special_tokens=True)
#
# # 打印输出结果
# print(output_text)