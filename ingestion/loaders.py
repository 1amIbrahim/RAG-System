from pathlib import Path
import fitz  # PyMuPDF
import docx

def load_pdf(path: Path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page": i + 1,
                "text": text
            })
    return pages

def load_docx(path: Path):
    doc = docx.Document(path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"page": 1, "text": full_text}]

def load_txt(path: Path):
    text = path.read_text(encoding="utf-8")
    return [{"page": 1, "text": text}]

def load_document(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".docx":
        return load_docx(path)
    elif suffix == ".txt":
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
