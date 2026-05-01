from PyPDF2 import PdfReader


def load_pdf_text(pdf_path: str) -> str:
    """
    Extract raw text from PDF
    """
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    return full_text
