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

import getpass
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

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
def split_text(txt,chunk_size=300,overlap=30,directory_path =r'E:\llm\deepseek\document_hc'):
    if not txt:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = overlap)
    docs = splitter.split_documents(data_loader(directory_path))

    # docs = splitter.split_text(txt)
    return docs


def create_embeddings(model_name="bge-m3:567m"):
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings


# def load_llm_ollama(model_path_name = "deepseek-r1:7b"):
#     chat_model = OllamaEmbeddings(model=model_path_name)
#     return chat_model

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
def load_or_create_vector_store(store_path,doc_file_path,model_name):
    embeddings = create_embeddings(model_name)
    vector_store = load_vector_store(store_path,embeddings)
    if not vector_store:
        docs = split_text(doc_file_path)
        vector_store = create_vector_store(docs,embeddings,store_path)
    return vector_store


# 从vector store查询上下文
def query_vector_store(vector_store, query, k=4, relevance_threshold=0.2):
    #len(vectorstore.similarity_search_with_relevance_scores(question, k=4)) 4
    #每条是一个Document 和一个分数
    # vectorstore.similarity_search_with_relevance_scores(question, k=4)[0][0].page_content 为内容
    similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    related_docs = list(filter(lambda x: x[1] > relevance_threshold, similar_docs))
    context = [doc[0].page_content for doc in related_docs]
    # context = ["\n\n".join(doc[0].page_content) for doc in related_docs]
    return context


# 加载 llm(deepseek) and tokenizer
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

def ollama_load_llm(temperature = 0.6,num_predict=512,model_name="deepseek-r1:7b"):
    model = ChatOllama(
        model=model_name,
        temperature = temperature,
        num_predict = num_predict
    )
    return model


def format_docs(docs):
    return "\n\n".join(doc for doc in docs)

# def ask_chain_prompt():
#     load_or_create_vector_store






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
    store_path =  r'E:\llm\deepseek\Data_vecstore\Aquila.faiss'
    Embedding_Model = 'bge-m3:567m'
    # LLM_Model = r'E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B'
    LLM_Model_name = 'deepseek-r1:7b'
    # LLM_Model_path = r'E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B'
    CUDA_Device = 'cuda:0'
    model = ollama_load_llm(model_name=LLM_Model_name)
    vector_store = load_or_create_vector_store(store_path, doc_file_path,model_name=Embedding_Model)


    # model, tokenizer = load_llm(LLM_Model,CUDA_Device)
    # RAG_TEMPLATE = """
    # 您是问答任务的助手。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三句话并保持答案简洁
    # <context>
    # {context}
    # </context>
    # 回答以下问题:
    # {question}"""
    RAG_TEMPLATE = """
    您是问答任务的助手。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。
    <context>
    {context}
    </context>
    回答以下问题:
    {question}"""
    retriever = vector_store.as_retriever()
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    # docs = query_vector_store()
    while True:
        qiz = input('请输入您的问题: ')
        if qiz == 'bye' or qiz == 'exit':
            print('Bye~')
            break


        # Query context from vector store based on question, and compose prompt
        context = query_vector_store(vector_store, qiz, 6, 0.2)
        # prompt = ChatPromptTemplate.from_template(
        #     "总结这些检索到的文档中的主要主题和内容: {docs}"
        # )

        if len(context) == 0:
            # No satisfying context is found inside vector store
            print('无法从保存的向量存储中找到限定的上下文,在没有上下文的情况下与LLM交谈')
            # prompt = f'请回答以下问题: \n{qiz}\n'
            print(model.invoke(qiz))
        else:
            # context = '\n'.join(context)
            # context = format_docs(context)
            prompt = f'基于以下上下文: \n{context}\n请回答以下问题: \n{qiz}\n'
            prompt = ChatPromptTemplate.from_template(
                "总结这些检索到的文档中的主要主题和内容: {docs},基于以上上下文回答"
            )
            # chain = {"docs": format_docs} | prompt | model | StrOutputParser()
            # chain = (
            #         {"context": retriever | format_docs, "question": RunnablePassthrough()}
            #         | rag_prompt
            #         | model
            #         | StrOutputParser()
            # )
            context_str = '\n'.join(context)
            def get_context(_):
                return context_str
            chain = (
                    {"context": get_context, "question": RunnablePassthrough()}
                    | rag_prompt
                    | model
                    | StrOutputParser()
            )
            print(chain.invoke(qiz))
            print(1)
        # ans = ask(model, tokenizer, prompt,CUDA_Device)[len(prompt):]
        # print(ans)

if __name__ == '__main__':
    main()
