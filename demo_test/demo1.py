from fastapi import FastAPI
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
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").half().to(device)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    top_p=1,
    repetition_penalty=1.15,
    truncation=True

)
llama_model = HuggingFacePipeline(pipeline=pipe)


msg = [
    SystemMessage(content='请将以下的内容翻译成意大利语'),
    HumanMessage(content='你好，请问你要去哪里？')
]

# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "请将下面的内容翻译为{language}"),
    ('user', "{text}")
])



# chain = LLMChain(llm=llama_model, prompt=prompt)
parser = StrOutputParser()
chain =   prompt_template |  parser | llama_model


print(chain.invoke({'language': 'English','text':'我下午还有一节课，不能去打球了。'}))
# return_str = parser.invoke(chain)
# print(return_str)
# print()
# print('chain.invoke("天津")',chain.invoke("天津"))



#把我们的程序部署成服务
#创建fastAPI的应用

app = FastAPI(title='千翔langchain服务',version='V1.0',description='使用Langchain服务')

add_routes