import os
import glob
import PyPDF2
import numpy as np
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def load_all_pdfs(folder_path):
    """Loads text from all PDF files in the specified folder."""
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    all_texts = {}
    for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
        text = extract_text_from_pdf(pdf_file)
        if text.strip():
            all_texts[pdf_file] = text
    return all_texts

def split_text(text, max_tokens=500):
    """
    Splits text into chunks of roughly max_tokens.
    This simple splitter uses newline as a separator.
    """
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len((current_chunk + " " + para).split()) < max_tokens:
            current_chunk += " " + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def cosine_similarity(a, b):
    """Computes cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)