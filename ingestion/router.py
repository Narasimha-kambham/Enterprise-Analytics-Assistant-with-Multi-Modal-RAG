from ingestion.pdf_ingestion import ingest_pdf
from ingestion.csv_ingestion import ingest_csv_folder
from ingestion.text_ingestion import ingest_text_folder
from ingestion.image_ingestion import ingest_image_folder
from config.settings import PDF_DIR, CSV_DIR, TEXT_DIR, IMAGE_DIR
import os

def ingest_all(base_path="data"):
    all_docs = []

    print(f"\n🚀 Starting Ingestion from '{base_path}' folder...")
    print("-" * 50)

    # PDFs
    pdf_folder = PDF_DIR if os.path.exists(PDF_DIR) else os.path.join(base_path, "pdfs")
    if os.path.exists(pdf_folder):
        print(f"📁 Processing PDFs...")
        pdf_docs_count = 0
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                print(f"  📄 Ingesting: {file}...")
                docs = ingest_pdf(os.path.join(pdf_folder, file))
                all_docs.extend(docs)
                pdf_docs_count += len(docs)
        print(f"✔️  PDF ingestion successful. Created {pdf_docs_count} documents.")
    else:
        print(f"⚠️  PDF folder not found: {pdf_folder}")

    # CSV
    csv_folder = CSV_DIR if os.path.exists(CSV_DIR) else os.path.join(base_path, "csv")
    if os.path.exists(csv_folder):
        print(f"📁 Processing CSVs...")
        csv_docs = ingest_csv_folder(csv_folder)
        all_docs.extend(csv_docs)
        print(f"✔️  CSV ingestion successful. Created {len(csv_docs)} documents.")
    else:
        print(f"⚠️  CSV folder not found: {csv_folder}")

    # Text
    text_folder = TEXT_DIR if os.path.exists(TEXT_DIR) else os.path.join(base_path, "text")
    if os.path.exists(text_folder):
        print(f"📁 Processing Text files...")
        text_docs = ingest_text_folder(text_folder)
        all_docs.extend(text_docs)
        print(f"✔️  Text ingestion successful. Created {len(text_docs)} documents.")
    else:
        print(f"⚠️  Text folder not found: {text_folder}")

    # Images
    image_folder = IMAGE_DIR if os.path.exists(IMAGE_DIR) else os.path.join(base_path, "images")
    if os.path.exists(image_folder):
        print(f"📁 Processing Images...")
        image_docs = ingest_image_folder(image_folder)
        all_docs.extend(image_docs)
        print(f"✔️  Image ingestion successful. Created {len(image_docs)} documents.")
    else:
        print(f"⚠️  Image folder not found: {image_folder}")

    print("-" * 50)
    print(f"✅ Ingestion Complete! Total documents created: {len(all_docs)}\n")

    return all_docs
