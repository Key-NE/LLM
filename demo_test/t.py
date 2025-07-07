import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_current_temperature(location: str, unit: str) -> float:
  """
  Get the current temperature at a location.

  Args:
      location: The location to get the temperature for, in the format "City, Country"
      unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
  Returns:
      The current temperature at the specified location in the specified units, as a float.
  """
  return 22.  # A real function should probably actually get the temperature!


def get_current_wind_speed(location: str) -> float:
  """
  Get the current wind speed in km/h at a given location.

  Args:
      location: The location to get the temperature for, in the format "City, Country"
  Returns:
      The current wind speed at the given location in km/h, as a float.
  """
  return 6.  # A real function should probably actually get the wind speed!


tools = [get_current_temperature, get_current_wind_speed]



tokenizer = AutoTokenizer.from_pretrained( r'E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B')
model = AutoModelForCausalLM.from_pretrained( r'E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B', torch_dtype=torch.bfloat16, device_map="auto")



messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]


inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
print(type(inputs),inputs)
inputs = {k: v for k, v in inputs.items()}
print(type(inputs),inputs)
outputs = model.generate(**inputs, max_new_tokens=128)
print(type(outputs),outputs)
print(tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):]))



#现在，将 get_current_temperature 函数和这些参数作为 tool_call 附加到聊天消息中。 应将 tool_call 字典提供给 assistant 角色，而不是系统或用户 。
# tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
# messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
# inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
# inputs = {k: v for k, v in inputs.items()}
# out = model.generate(**inputs, max_new_tokens=128)
# print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))



