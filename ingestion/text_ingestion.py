import os
from langchain_community.document_loaders import TextLoader

def ingest_text_folder(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            print(f"    📝 Reading text file: {file}...")
            loader = TextLoader(
                os.path.join(folder_path, file),
                encoding="utf-8"
            )
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = "text"
                doc.metadata["file_name"] = file

            documents.extend(docs)

    return documents
