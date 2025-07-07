from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain import  PromptTemplate
import torch
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage


import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention.")


model_path = r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-1.5B"
device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").half()
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    top_p=1,
    repetition_penalty=1.15,
    truncation=True

)

msg = [
    SystemMessage(content='请将以下的内容翻译成意大利语'),
    HumanMessage(content='你好，请问你要去哪里？')
]

#定义提示模板
prompt_template = ChatPromptTemplate.format_messages([
    ("system", "请将下面的内容翻译为{language}"),
    ('user',"{text}")
])


llama_model = HuggingFacePipeline(pipeline=pipe)
template = '''
你是一名知识丰富的导航助手，了解中国每一个地方的名胜古迹及旅游景点. 
游客:我想去{地方}旅游，给我推荐一下值得玩的地方?"
'''
prompt = PromptTemplate(
    input_variables=["地方"],
    template=template
)
# chain = LLMChain(llm=llama_model, prompt=prompt)
parser = StrOutputParser()
chain = prompt | llama_model | parser
# chain = llama_model | parser

result = chain.invoke("天津")
print('result',result)

return_str = parser.invoke(template)
# print(return_str)
# print()
# print('chain.invoke("天津")',chain.invoke("天津"))