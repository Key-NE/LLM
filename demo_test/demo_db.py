from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained(r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-1.5B")

# 输入文本
input_text = "你好"

# 对输入文本进行分词
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 生成文本
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)