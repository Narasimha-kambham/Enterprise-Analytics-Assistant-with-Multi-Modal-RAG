import os
import re
import fitz
import tabula
import numpy as np
import cv2
import pytesseract
from pathlib import Path
from pdf2image import convert_from_path
from langchain_core.documents import Document
from llm.vision import image_to_description
from config.settings import TESSERACT_PATH, POPPLER_PATH, EXTRACTED_IMAGES_DIR, TEXT_DENSITY_THRESHOLD

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Text Density

def calculate_alphabetic_characters(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text))


def is_text_sufficient(text: str, threshold: int = TEXT_DENSITY_THRESHOLD) -> bool:
    return calculate_alphabetic_characters(text) >= threshold

# Sampling and scanned detection

def is_scanned(pdf_path: str, threshold: int = TEXT_DENSITY_THRESHOLD) -> bool:
    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        if total_pages == 0:
            raise ValueError("PDF has no pages.")

        sample_indices = {0, total_pages // 2, total_pages - 1}

        for index in sample_indices:
            page = doc.load_page(index)
            text = page.get_text() or ""
            if is_text_sufficient(text, threshold):
                return False

        return True
    
# Document Creator

def create_document(text, pdf_name, page_number,
                    content_type, extraction_method):
    return Document(
        page_content=text,
        metadata={
            "source": "pdf",
            "pdf_name": pdf_name,
            "page": page_number,
            "content_type": content_type,
            "extraction_method": extraction_method
        }
    )

def ocr_page(pdf_path: str, page_number: int) -> str:
    images = convert_from_path(
        pdf_path,
        dpi=300,
        first_page=page_number + 1,
        last_page=page_number + 1,
        poppler_path=POPPLER_PATH
    )

    np_img = np.array(images[0])
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 6"
    )

    return text


def extract_tables_from_page(pdf_path: str, page_number: int):
    try:
        tables = tabula.read_pdf(
            pdf_path,
            pages=page_number + 1,
            multiple_tables=True,
            stream=True,
            encoding="cp1252"
        )
    except Exception as e:
        print(f"Tabula table extraction error: {e}")
        return []
    

    processed_tables = []

    for df in tables:
        if df is None or df.empty:
            continue

        df = df.fillna("")
        df = df.loc[~(df.astype(str).eq("").all(axis=1))]
        df.columns = [
            str(col).strip()
            if "Unnamed" not in str(col)
            else ""
            for col in df.columns
        ]

        df = df.astype(str)
        table_text = df.to_markdown(index=False)
        processed_tables.append(table_text)

    return processed_tables

def extract_images_from_page(pdf_path: str, page_number: int):
    output_dir = Path(EXTRACTED_IMAGES_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = Path(pdf_path).stem
    pdf_folder = output_dir / pdf_name
    pdf_folder.mkdir(parents=True, exist_ok=True)

    image_paths = []

    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_number)
        images = page.get_images(full=True)

        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")

            filename = pdf_folder / \
                f"Image_{page_number:03d}_{img_idx:02d}.{image_ext}"

            with open(filename, "wb") as f:
                f.write(image_bytes)

            image_paths.append(str(filename))

    return image_paths

def process_mixed_pdf(pdf_path: str, threshold: int = TEXT_DENSITY_THRESHOLD) -> list:
    documents = []
    pdf_name = Path(pdf_path).name

    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        print(f"    📄 Processing {total_pages} pages...")

        for page_index in range(total_pages):
            print(f"      - Page {page_index + 1}/{total_pages}...", end="\r")
            page = doc.load_page(page_index)
            native_text = page.get_text() or ""
            if is_text_sufficient(native_text, threshold):
                documents.append(
                    create_document(
                        text=native_text,
                        pdf_name=pdf_name,
                        page_number=page_index,
                        content_type="text",
                        extraction_method="native"
                    )
                )
            else:
                ocr_text = ocr_page(pdf_path, page_index)
                documents.append(
                    create_document(
                        text=ocr_text,
                        pdf_name=pdf_name,
                        page_number=page_index,
                        content_type="ocr_text",
                        extraction_method="tesseract"
                    )
                )
            # Table extraction
            tables = extract_tables_from_page(pdf_path, page_index)
            for table in tables:
                documents.append(
                    create_document(
                        text=table,
                        pdf_name=pdf_name,
                        page_number=page_index,
                        content_type="table",
                        extraction_method="tabula"
                    )
                )
            # Image extraction
            image_paths = extract_images_from_page(pdf_path, page_index)
            for image_path in image_paths:
                image_description = image_to_description(image_path)
                if image_description.strip():
                    documents.append(
                        create_document(
                            text=image_description,
                            pdf_name=pdf_name,
                            page_number=page_index,
                            content_type="image_description",
                            extraction_method="vision"
                        )
                    )
        print(f"      - Page {total_pages}/{total_pages}... Done.         ")
    return documents


def process_scanned_pdf(pdf_path: str, threshold: int = TEXT_DENSITY_THRESHOLD) -> list:
    """
    Process a fully scanned PDF using consistent OCR preprocessing.
    Applies OCR per page with image enhancement.
    """

    documents = []
    file_name = os.path.basename(pdf_path)
    pdf_name = Path(pdf_path).stem

    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        print(f"    📄 Processing {total_pages} scanned pages...")

        for page_number in range(total_pages):
            print(f"      - Page {page_number + 1}/{total_pages} (OCR)...", end="\r")
            # 1️⃣ OCR page using standardized pipeline
            ocr_text = ocr_page(pdf_path, page_number)

            # 2️⃣ Store if sufficient
            if is_text_sufficient(ocr_text, threshold):

                documents.append(
                    create_document(
                        text=ocr_text,
                        pdf_name=pdf_name,
                        page_number=page_number,
                        content_type="ocr_text",
                        extraction_method="tesseract"
                    )
                )
        print(f"      - Page {total_pages}/{total_pages}... Done.         ")

    return documents

def ingest_pdf(pdf_path: str, threshold: int = TEXT_DENSITY_THRESHOLD):
    if is_scanned(pdf_path, threshold):
        print("    🔍 Scanned PDF detected → Using OCR pipeline")
        return process_scanned_pdf(pdf_path, threshold)
    else:
        print("    🔍 Native/Mixed PDF detected → Using standard pipeline")
        return process_mixed_pdf(pdf_path, threshold)

