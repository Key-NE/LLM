from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_ollama import ChatOllama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from langchain.document_loaders import TextLoader,Docx2txtLoader
from PyPDF2 import PdfReader

from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch

# Embedding_Model = 'BAAI/bge-large-zh'
Embedding_Model = 'bge-m3'
# Embedding_Model_path = r'E:\llm\deepseek\BAAIbge-large-zh'
LLM_Model = 'DeepSeek-R1-Distill-Qwen-1.5B'
# LLM_Model_path = r'E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B'
CUDA_Device = 'cuda:0'


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
def split_text(txt,chunk_size=1000,overlap=100):
    if not txt:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = overlap)
    docs = splitter.split_text(txt)
    return docs




def create_embeddings():
    embeddings = OllamaEmbeddings(model="bge-m3")
    return embeddings


# 使用Embeddings嵌入模型将文档保存到向量知识库存储
def create_vector_store(docs, embeddings, store_path):
    vector_store = FAISS.from_texts(docs,embeddings)
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
def load_or_create_vector_store(store_path,pdf_file_path):
    embeddings = create_embeddings()
    vector_store = load_vector_store(store_path,embeddings)
    if not vector_store:
        txt = load_single_pdf(pdf_file_path)
        docs = split_text(txt)
        vector_store = create_vector_store(docs,embeddings,store_path)
    return vector_store


# 从vector store查询上下文
def query_vector_store(vector_store, query, k=4, relevance_threshold=0.8):
    similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    related_docs = list(filter(lambda x: x[1] > relevance_threshold, similar_docs))
    context = [doc[0].page_content for doc in related_docs]
    return context


# Load llm(deepseek) and tokenizer
def load_llm(model_path):
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


def ask(model, tokenizer, prompt, max_tokens=512):
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






splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
split = splitter.split_documents(data_loader(directory_path=r'E:\llm\deepseek\document_hc'))


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# all_splits = text_splitter.split_documents(data)

# print(all_splits)


local_embeddings = OllamaEmbeddings(model="bge-m3")

vectorstore = Chroma.from_documents(documents=split, embedding=local_embeddings)


# question = "What are the approaches to Task Decomposition?"
question = "入场工作时的安全注意事项是什么？"
docs = vectorstore.similarity_search(question)

# print('len(docs)',len(docs))
# print(docs[0])
# print(type(docs[0]))
# print(type(docs))
# docs[0].page_content

model = ChatOllama(
    model="deepseek-r1:7b",
)

# response_message = model.invoke(
#     "Simulate a rap battle between Stephen Colbert and John Oliver"
# )

# print(response_message.content)

# prompt = ChatPromptTemplate.from_template(
#     "Summarize the main themes in these retrieved docs: {docs}"
# )
prompt = ChatPromptTemplate.from_template(
    "总结这些检索到的文档中的主要主题和内容: {docs}"
)


# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = {"docs": format_docs} | prompt | model | StrOutputParser()

# question = "What are the approaches to Task Decomposition?"
question = "入场工作时的安全注意事项是什么？"
docs = vectorstore.similarity_search(question)

# print(chain.invoke(docs))
#


# RAG_TEMPLATE = """
# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#
# <context>
# {context}
# </context>
#
# Answer the following question:
#
# {question}"""

RAG_TEMPLATE = """
您是问答任务的助手。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三句话并保持答案简洁
<context>
{context}
</context>
回答以下问题:
{question}"""




rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

chain = (
    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    | rag_prompt
    | model
    | StrOutputParser()
)

question = "What are the approaches to Task Decomposition?"

docs = vectorstore.similarity_search(question)

# Run
chain.invoke({"context": docs, "question": question})


retriever = vectorstore.as_retriever()

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

question = "What are the approaches to Task Decomposition?"

qa_chain.invoke(question)


# print(qa_chain.invoke(question))




def main():
    doc_file_path = r'E:\llm\deepseek\document_hc'
    store_path = './Data/Aquila.faiss'

    vector_store = load_or_create_vector_store(store_path, doc_file_path)
    model, tokenizer = load_llm(LLM_Model)

    while True:
        qiz = input('Please input question: ')
        if qiz == 'bye' or qiz == 'exit':
            print('Bye~')
            break

        # Query context from vector store based on question, and compose prompt
        context = query_vector_store(vector_store, qiz, 6, 0.75)
        if len(context) == 0:
            # No satisfying context is found inside vector store
            print('Cannot find qualified context from the saved vector store. Talking to LLM without context.')
            prompt = f'Please answer the question: \n{qiz}\n'
        else:
            context = '\n'.join(context)
            prompt = f'Based on the following context: \n{context}\nPlease answer the question: \n{qiz}\n'

        ans = ask(model, tokenizer, prompt)[len(prompt):]
        print(ans)

if __name__ == '__main__':
    main()
