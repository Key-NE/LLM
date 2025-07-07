import os
from docx import Document


class ReadFiles:
    def __init__(self, path: str):
        self.path = path

    def list_files(self):
        file_list = []
        for filepath, _, filenames in os.walk(self.path):
            for filename in filenames:
                file_list.append(os.path.join(filepath, filename))
        return file_list

    def read_file_content(self, file_path: str):
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.txt') or file_path.endswith('.stt'):
            return self.read_text(file_path)
        elif file_path.endswith('.docx'):
            return self.read_docx(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return None

    def read_text(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def read_docx(self, file_path: str):
        doc = Document(file_path)
        contents = [para.text for para in doc.paragraphs]
        return "\n\n".join(contents)

    def split_chunks(self, text: str):
        return text.split("\n\n")

    def load_content(self):
        docs = []
        file_list = self.list_files()

        for file_path in file_list:
            # 读取文件内容
            content = self.read_file_content(file_path)
            if content is None:
                continue
            docs.extend(self.split_chunks(content))

        return docs
















# 使用示例
if __name__ == "__main__":
    path_to_files = 'E:/llm/deepseek/document_hc/'
    reader = ReadFiles(path_to_files)
    content = reader.load_content()
    for doc in content:
        # print(1)
        print(doc)