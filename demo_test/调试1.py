import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama



os.environ["http_proxy"] = "http://127.0.0.1:11434"
os.environ["https_proxy"] = "http://127.0.0.1:11434"
model = ChatOllama(
    model="deepseek-r1:7b",
)
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]

# model.invoke(messages)
print(model.invoke(messages))

# from langchain_core.runnables import RunnablePassthrough
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
#
# rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
#
# chain = (
#     RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
#     | rag_prompt
#     | model
#     | StrOutputParser()
# )
#
# question = "What are the approaches to Task Decomposition?"
#
# docs = vectorstore.similarity_search(question)
#
# # Run
# chain.invoke({"context": docs, "question": question})
#
#
# retriever = vectorstore.as_retriever()
#
# qa_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | rag_prompt
#     | model
#     | StrOutputParser()
# )
#
# question = "What are the approaches to Task Decomposition?"
#
# qa_chain.invoke(question)