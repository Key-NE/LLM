from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_ollama import ChatOllama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
# from langchain.document_loaders import TextLoader,Docx2txtLoader
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from PyPDF2 import PdfReader

from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch

os.environ["http_proxy"] = "http://127.0.0.1:11434"
os.environ["https_proxy"] = "http://127.0.0.1:11434"


#加载pdf文件并返回文本段
def load_single_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    if not pdf_reader:
        return None
    ret = ''
    for i,page in enumerate(pdf_reader.pages):
        txt = page.extractText()
        if txt:
            ret += txt
        return ret


#文档加载器
def data_loader(directory_path=r'E:\llm\deepseek\document_hc'):
    all_docs = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif file.endswith('.pdf'):
                loader = load_single_pdf(file_path)
            else:
                continue
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs


# 将文本拆分为 docs文档
def split_text(txt,chunk_size=1000,overlap=100,directory_path =r'E:\llm\deepseek\document_hc'):
    if not txt:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = overlap)
    docs = splitter.split_documents(data_loader(directory_path))

    # docs = splitter.split_text(txt)
    return docs


def create_embeddings():
    embeddings = OllamaEmbeddings(model="bge-m3:567m")
    return embeddings


def load_llm_ollama(model_path_name = "deepseek-r1:7b"):
    chat_model = OllamaEmbeddings(model=model_path_name)
    return chat_model

# 使用Embeddings嵌入模型将文档保存到向量知识库存储
def create_vector_store(docs, embeddings, store_path):
    # vectorstore = Chroma.from_documents(documents=split, embedding=local_embeddings)
    # vector_store = FAISS.from_texts(docs,embeddings)

    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(store_path)
    return vector_store


# 从文件加载向量知识库
def load_vector_store(store_path,embeddings):
    if os.path.exists(store_path):
        vector_store = FAISS.load_local(
            store_path,
            embeddings = embeddings,
            allow_dangerous_deserialization=True
                                        )
        return vector_store
    else:
        return None



# 加载或者创建向量知识库
def load_or_create_vector_store(store_path,doc_file_path):
    embeddings = create_embeddings()
    vector_store = load_vector_store(store_path,embeddings)
    if not vector_store:
        docs = split_text(doc_file_path)
        vector_store = create_vector_store(docs,embeddings,store_path)
    return vector_store


# 从vector store查询上下文
def query_vector_store(vector_store, query, k=4, relevance_threshold=0.3):
    similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    related_docs = list(filter(lambda x: x[1] > relevance_threshold, similar_docs))
    context = [doc[0].page_content for doc in related_docs]
    return context


# Load llm(deepseek) and tokenizer
def load_llm(model_path,CUDA_Device):
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
        device_map=CUDA_Device,
        torch_dtype=torch.float16,
        quantization_config=quant_config)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path,
        use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer




def ask(model, tokenizer, prompt, CUDA_Device,max_tokens=512 ):
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids('<|eot_id|>')
    ]
    input_ids = tokenizer([prompt],
        return_tensors='pt',
        add_special_tokens=False).input_ids.to(CUDA_Device)
    generated_input = {
        'input_ids': input_ids,
        'max_new_tokens': max_tokens,
        'do_sample': True,
        'top_p': 0.95,
        'temperature': 0.9,
        'repetition_penalty': 1.1,
        'eos_token_id': terminators,
        'bos_token_id': tokenizer.bos_token_id,
        'pad_token_id': tokenizer.pad_token_id
    }

    generated_ids = model.generate(**generated_input)
    ans = tokenizer.decode(generated_ids[0], skip_special_token=True)
    return ans



def main():
    # 初始化
    doc_file_path = r'E:\llm\deepseek\document_hc'
    # store_path = './Data/Aquila.faiss'
    store_path =  r'E:\llm\deepseek\Data_vecstore\Aquila.faiss'
    Embedding_Model = 'bge-m3:567m'
    LLM_Model = r'E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B'
    # LLM_Model_path = r'E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B'
    CUDA_Device = 'cuda:0'

    vector_store = load_or_create_vector_store(store_path, doc_file_path)
    model, tokenizer = load_llm(LLM_Model,CUDA_Device)

    while True:
        qiz = input('请输入您的问题: ')
        if qiz == 'bye' or qiz == 'exit':
            print('Bye~')
            break


        # Query context from vector store based on question, and compose prompt
        context = query_vector_store(vector_store, qiz, 6, 0.75)
        if len(context) == 0:
            # No satisfying context is found inside vector store
            print('无法从保存的向量存储中找到限定的上下文,在没有上下文的情况下与LLM交谈')
            prompt = f'请回答以下问题: \n{qiz}\n'
        else:
            context = '\n'.join(context)
            prompt = f'基于以下上下文: \n{context}\n请回答以下问题: \n{qiz}\n'

        ans = ask(model, tokenizer, prompt,CUDA_Device)[len(prompt):]
        print(ans)

if __name__ == '__main__':
    main()
