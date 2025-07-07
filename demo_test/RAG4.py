from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline, AutoModelForCausalLM,GenerationConfig
from transformers import AutoTokenizer, AutoModel
from langchain_chroma import Chroma
from transformers import BertTokenizer, BertModel
import torch
from langchain.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
import ollama
from ollama import ChatResponse
from langchain.llms import HuggingFacePipeline
# from langchain_deepseek import ChatDeepSeek

def load_embedding_mode(self,model_name='text2vec3'):
    model_path = self.embedding_model_dict[model_name]
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path).to("cuda:0")

    def embedding_function(text):
        inputs = tokenizer(text, return_tensors="pt",padding=True,truncation=True,max_length=512).to("cuda:0")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
        return embeddings

class EmbeddingFunction:
    def __init__(self,embedding_function):
        self.embedding_function = embedding_function

    def embed_query(self,query):
        return self.embedding_function(query)

    def embed_documents(self,documents):
        return [self.embedding_function(doc) for doc in documents]


tokenizer_chat = AutoTokenizer.from_pretrained(r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-1.5B")
model_chat = AutoModelForCausalLM.from_pretrained(r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-1.5B")
model_chat  = ChatDeepSeek(
    model=r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-1.5B",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",
    # other params...
)
# generation_config = GenerationConfig(
#     max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
# )
# api_key = "EMPTY"
# api_url = "http://0.0.0.0:8020/v1"
# model = "deepseek"
# model_chat = ChatOpenAI(model_name=model, openai_api_key=api_key, openai_api_base=api_url)
loader = PyPDFLoader("https://arxiv.org/pdf/2402.03216")

docs = loader.load()

print(docs[0].metadata)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Maximum size of chunks to return
    chunk_overlap=150,  # number of overlap characters between chunks
)
corpus = splitter.split_documents(docs)
print(corpus)


# embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5",
# encode_kwargs={"normalize_embeddings": True})


# tokenizer = AutoTokenizer.from_pretrained(r"E:\llm\deepseek\BAAIbge-large-zh")
# model = AutoModel.from_pretrained(r"E:\llm\deepseek\BAAIbge-large-zh")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    cache_folder=r"E:\llm\deepseek\BAAIbge_large_zh",
    encode_kwargs={"normalize_embeddings": True}
)
vectordb = FAISS.from_documents(corpus, embedding_model)

# vectorstore = Chroma.from_documents(documents=corpus,embedding=embedding_model,persist_directory="./db2")
# print(vectorstore)
# print('1111---------------------------')
vectordb.save_local("vectorstore.db")
retriever = vectordb.as_retriever()

template = """
You are a Q&A chat bot.
Use the given context only, answer the question.

<context>
{context}
</context>

Question: {input}
"""

prompt = ChatPromptTemplate.from_template(template)
doc_chain = create_stuff_documents_chain(model_chat, prompt)
chain = create_retrieval_chain(retriever, doc_chain)
response = chain.invoke({"input": "What does M3-Embedding stands for?"})

# print the answer only
# print(response['answer'])