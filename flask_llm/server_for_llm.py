from flask import Flask, jsonify, request
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from PyPDF2 import PdfReader
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


class LLMService:
    def __init__(self):
        self.doc_file_path = r'E:\llm\deepseek\document_hc'
        self.store_path = r'E:\llm\deepseek\Data_vecstore\Aquila.faiss'
        self.Embedding_Model = 'nomic-embed-text'
        self.LLM_Model_name = 'deepseek-r1:14b'
        self.CUDA_Device = 'cuda:0'
        self.model = self.ollama_load_llm()
        self.vector_store = self.load_or_create_vector_store()

    # 加载pdf文件并返回文本段
    def load_single_pdf(self, file_path):
        pdf_reader = PdfReader(file_path)
        if not pdf_reader:
            return None
        ret = ''
        for page in pdf_reader.pages:
            txt = page.extract_text()
            if txt:
                ret += txt
        return ret

    # 文档加载器
    def data_loader(self):
        all_docs = []
        for root, dirs, files in os.walk(self.doc_file_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                elif file.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                elif file.endswith('.pdf'):
                    txt = self.load_single_pdf(file_path)
                    if txt:
                        from langchain.docstore.document import Document
                        doc = Document(page_content=txt)
                        all_docs.append(doc)
                    continue
                else:
                    continue
                docs = loader.load()
                all_docs.extend(docs)
        return all_docs

    # 将文本拆分为docs文档
    def split_text(self, chunk_size=300, overlap=30):
        txt = self.data_loader()
        if not txt:
            return None
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap)
        docs = splitter.split_documents(txt)
        return docs

    def create_embeddings(self):
        embeddings = OllamaEmbeddings(model=self.Embedding_Model)
        return embeddings

    # 使用Embeddings嵌入模型将文档保存到向量知识库存储
    def create_vector_store(self, docs, embeddings):
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(self.store_path)
        return vector_store

    # 从文件加载向量知识库
    def load_vector_store(self, embeddings):
        if os.path.exists(self.store_path):
            vector_store = FAISS.load_local(
                self.store_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        else:
            return None

    # 加载或者创建向量知识库
    def load_or_create_vector_store(self):
        embeddings = self.create_embeddings()
        vector_store = self.load_vector_store(embeddings)
        if not vector_store:
            docs = self.split_text()
            vector_store = self.create_vector_store(docs, embeddings)
        return vector_store

    # 从vector store查询上下文
    def query_vector_store(self, query, k=4, relevance_threshold=0.2):
        similar_docs = self.vector_store.similarity_search_with_relevance_scores(
            query, k=k)
        related_docs = list(
            filter(lambda x: x[1] > relevance_threshold, similar_docs))
        context = [doc[0].page_content for doc in related_docs]
        return context

    def ollama_load_llm(self, temperature=0.6, num_predict=512):
        model = ChatOllama(
            model=self.LLM_Model_name,
            temperature=temperature,
            num_predict=num_predict
        )
        return model

    def format_docs(self, docs):
        return "\n\n".join(doc for doc in docs)

    def llm_invoke(self, qiz):
        RAG_TEMPLATE = """
        您是问答任务的助手。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。
        <context>
        {context}
        </context>
        回答以下问题:
        {question}"""
        retriever = self.vector_store.as_retriever()
        rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

        context = self.query_vector_store(qiz, 6, 0.2)

        if len(context) == 0:
            return self.model.invoke(qiz)
        else:
            def get_context(_):
                return self.format_docs(context)

            chain = (
                    {"context": get_context, "question": RunnablePassthrough()}
                    | rag_prompt
                    | self.model
                    | StrOutputParser()
            )
            return chain.invoke(qiz)


app = Flask(__name__)
llm_service = LLMService()


@app.route('/api/llm_invoke', methods=['POST'])
def get_answer():
    data = request.get_json()
    if 'question' not in data:
        return jsonify({"error": "请求缺少question参数"}), 400

    question = data['question']
    answer = llm_service.llm_invoke(question)
    return jsonify({"answer": str(answer)}), 200


if __name__ == '__main__':
    os.environ["http_proxy"] = "http://127.0.0.1:11434"
    os.environ["https_proxy"] = "http://127.0.0.1:11434"
    app.run(host='0.0.0.0', port=5010, debug=True)
