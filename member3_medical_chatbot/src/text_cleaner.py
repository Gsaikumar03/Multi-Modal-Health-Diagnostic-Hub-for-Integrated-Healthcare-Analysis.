import re


def clean_text(text: str) -> str:
    """
    Basic text cleaning
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500) -> list:
    """
    Split text into fixed-size chunks
    """
    words = text.split(" ")
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
