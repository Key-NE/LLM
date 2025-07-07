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

model = ChatOllama(
    model="deepseek-r1:7b",
)


prompt = ChatPromptTemplate.from_template(
    "总结这些检索到的文档中的主要主题和内容: {docs}"
)


# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = {"docs": format_docs} | prompt | model | StrOutputParser()

question = "入场工作时的安全注意事项是什么？"
docs = vectorstore.similarity_search(question)

# print(chain.invoke(docs))


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

question = "任务分解的方法有哪些？"
qa_chain.invoke(question)


# print(qa_chain.invoke(question))