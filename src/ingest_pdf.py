from pathlib import Path
from tqdm import tqdm
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


PDF_PATH = Path("data/code_education.pdf")
DB_DIR = Path("db/faiss_code_edu_pdf")

def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i in tqdm(range(len(reader.pages)), desc="Extract PDF"):
        t = reader.pages[i].extract_text() or ""
        # nettoyage simple
        t = " ".join(t.split())
        if t:
            pages.append(f"[PAGE {i+1}] {t}")
    return "\n\n".join(pages)

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF introuvable: {PDF_PATH}")

    text = extract_pdf_text(PDF_PATH)
    if len(text) < 5000:
        raise RuntimeError("Texte extrait trop faible. Ton PDF est peut-être scanné -> OCR nécessaire.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_texts(
        chunks,
        embedding=embeddings,
        metadatas=[{"source": "code_education_pdf"} for _ in chunks],
    )

    DB_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(DB_DIR))
    print(f"OK: index créé: {DB_DIR} | chunks={len(chunks)}")

if __name__ == "__main__":
    main()
