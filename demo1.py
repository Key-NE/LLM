# from local_huggingface_embeddings import HuggingFaceEmbeddings

from .demo_test.huggingface_demo import HuggingFaceEmbeddings
model_name = r"E:\llm\deepseek\BAAIbge-large-zh"  # 替换为本地模型的实际路径
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

texts = ["这是一个测试文本", "另一个测试文本"]
embeddings = hf.embed_documents(texts)
print(embeddings)