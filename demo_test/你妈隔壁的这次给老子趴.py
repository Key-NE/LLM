import os
import torch
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from langchain.embeddings import HuggingFaceBgeEmbeddings
from FlagEmbedding import FlagModel
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings


Embedding_Model = 'BAAI/bge-large-zh'
Embedding_Model_path = r'E:\llm\deepseek\BAAIbge-large-zh'
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


# 将文本拆分为 docs文档
def split_text(txt,chunk_size=256,overlap=32):
    if not txt:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = overlap)
    docs = splitter.split_text(txt)
    return docs



#加载本地词嵌入模型   BAAIbge-large-zh
def create_embeddings():
    # embeddings = HuggingFaceEmbeddings(
    #     #     model_name = Embedding_Model,
    #     #     model_kwargs = {'device': CUDA_Device},
    #     #     cache_folder = Embedding_Model_path
    #     # )
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


def main():
    pdf_file_path = r'E:\llm\deepseek\document_hc'
    store_path = './Data/Aquila.faiss'

    vector_store = load_or_create_vector_store(store_path, pdf_file_path)
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












