from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained( r'E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B')
model = AutoModelForCausalLM.from_pretrained( r'E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B', torch_dtype=torch.bfloat16, device_map="auto")

device = model.device # Get the device the model is loaded on

# Define conversation input
conversation = [
    {"role": "user", "content": "What has Man always dreamed of?"}
]


documents = [
    {
        "title": "The Moon: Our Age-Old Foe",
        "text": "Man has always dreamed of destroying the moon. In this essay, I shall..."
    },
    {
        "title": "The Sun: Our Age-Old Friend",
        "text": "Although often underappreciated, the sun provides several notable benefits..."
    }
]


input_ids = tokenizer.apply_chat_template(
    conversation=conversation,
    documents=documents,
    chat_template="rag",
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt").to(device)

# Generate a response
generated_tokens = model.generate(
    input_ids,
    max_new_tokens=1000,
    do_sample=True,
    temperature=0.6,
    )

# Decode and print the generated text along with generation prompt
generated_text = tokenizer.decode(generated_tokens[0])
print(generated_text)