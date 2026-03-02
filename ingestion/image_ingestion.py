import os
from langchain_core.documents import Document
from llm.vision import image_to_description

def ingest_image_folder(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder_path, file)
            print(f"    🖼️  Analyzing image: {file}...")

            description = image_to_description(path)

            if description.strip():
                documents.append(
                    Document(
                        page_content=description,
                        metadata={
                            "source": "image",
                            "file_name": file,
                            "extraction_method": "vision"
                        }
                    )
                )
    
    return documents
