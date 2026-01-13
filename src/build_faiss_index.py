#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_faiss_index.py

Construit un index FAISS à partir de data/chunks_articles.jsonl (article_id + text).

Sortie :
- db/faiss_code_edu_by_article (dossier FAISS LangChain)

Usage :
python src/build_faiss_index.py
"""

import json
from pathlib import Path

from tqdm import tqdm

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


CHUNKS_PATH = Path("data/chunks_articles.jsonl")
DB_DIR = Path("db/faiss_code_edu_by_article")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            article_id = obj["article_id"]
            text = obj["text"].strip()
            if len(text) < 30:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"article_id": article_id}
                )
            )
    return docs


def main():
    print(f"1) Lecture chunks : {CHUNKS_PATH}")
    docs = load_chunks(CHUNKS_PATH)
    print(f"OK: {len(docs)} chunks chargés")

    print("2) Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("3) Construction FAISS...")
    # LangChain gère le batching en interne, mais tqdm donne un ressenti
    # On construit directement depuis la liste de Documents.
    db = FAISS.from_documents(docs, embeddings)

    print(f"4) Sauvegarde index : {DB_DIR}")
    DB_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(DB_DIR))

    print("OK: index FAISS créé")


if __name__ == "__main__":
    main()
