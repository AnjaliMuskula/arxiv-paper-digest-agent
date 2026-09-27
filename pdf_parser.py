import fitz
import pytesseract
from PIL import Image
import io


MIN_READABLE_CHARS = 100


def extract_with_pymupdf(pdf_path: str):

    document = fitz.open(pdf_path)

    pages = []

    for page in document:
        text = page.get_text("text")
        pages.append(text)

    document.close()

    return "\n".join(pages)


def extract_with_ocr(pdf_path: str):

    document = fitz.open(pdf_path)

    pages = []

    for page in document:

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2)
        )

        image_bytes = pix.tobytes("png")

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        text = pytesseract.image_to_string(image)

        pages.append(text)

    document.close()

    return "\n".join(pages)


def parse_pdf(pdf_path: str):

    text = extract_with_pymupdf(pdf_path)

    cleaned_text = text.strip()

    if len(cleaned_text) < MIN_READABLE_CHARS:

        print("PDF text is empty/poor. Switching to OCR.")

        text = extract_with_ocr(pdf_path)

    else:

        print("PDF contains readable text. Using PyMuPDF.")

    return text


# PDF
#  │
#  ▼
# PyMuPDF
#  │
#  ├── readable ───────► use extracted text
#  │
#  └── empty/poor
#           │
#           ▼
#          OCR