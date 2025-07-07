from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# 1. 数据加载
loader = TextLoader('your_long_text.txt')
documents = loader.load()

# 2. 文本分割
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 3. 嵌入生成
embeddings = OpenAIEmbeddings()

# 4. 向量存储
db = FAISS.from_documents(docs, embeddings)

# 5. 检索器设置
retriever = db.as_retriever()

# 6. 提示工程
prompt_template = """使用以下上下文来回答问题。如果上下文没有提供足够的信息，请说“没有足够的信息来回答”。

{context}

问题: {question}
回答:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# 7. 推理流程
llm = OpenAI()
chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

def answer_question(question):
    relevant_docs = retriever.get_relevant_documents(question)
    answer = chain.run(input_documents=relevant_docs, question=question)
    return answer

# 测试问答
question = "你的长文本中关于某个特定主题的信息是什么？"
result = answer_question(question)
print(result)