#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chunk_by_article.py 

Objectif :
- Extraire le texte d'un PDF (Code de l'éducation)
- Supprimer le footer répétitif présent à chaque bas de page
- Découper en chunks en utilisant les en-têtes "Article ..."
- Un chunk = "Article X..." + tout le contenu jusqu'au prochain "Article Y..."

Sorties :
- data/chunks_articles.jsonl : 1 chunk par ligne (article_id + text)
- data/chunks_preview.md     : aperçu (EXTRAIT) pour vérifier visuellement
- data/chunks_audit.md       : rapport d'audit (chunks suspects / incomplets)

Remarques importantes :
- Le .md "preview" est volontairement un APERÇU : il tronque chaque chunk (par défaut 1200 caractères).
  Le JSONL, lui, contient le texte COMPLET du chunk.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from pypdf import PdfReader
from tqdm import tqdm


# -----------------------------------------------------------------------------
# 1) Nettoyage : suppression robuste du footer Légifrance
# -----------------------------------------------------------------------------
# Ton footer exact (tu l'as donné) :
# "Code de l'éducation - Dernière modification le 07 janvier 2026 - Document généré le 11 janvier 2026"
#
# Problème : selon l'extraction PDF, il peut apparaître :
# - en fin de ligne
# - collé à d'autres textes
# - avec des espaces multiples
#
# Stratégie :
# - on supprime TOUTE ligne qui commence par "Code de l'éducation - Dernière modification"
# - et/ou toute occurrence de "Document généré le <date>"
# en mode MULTILINE (ligne par ligne) + tolérance aux espaces.
FOOTER_LINE_RE = re.compile(
    r"(?im)^\s*Code\s+de\s+l['’]éducation\s*-\s*Dernière\s+modification\s+le\s+.*?\s*-\s*Document\s+généré\s+le\s+.*?\s*$"
)

# Au cas où le footer est injecté au milieu d'une ligne (ça arrive), on enlève aussi ces motifs
FOOTER_INLINE_RE = re.compile(
    r"(?i)Code\s+de\s+l['’]éducation\s*-\s*Dernière\s+modification\s+le\s+.*?\s*-\s*Document\s+généré\s+le\s+.*?(?=(\n|$))"
)


def clean_text(text: str) -> str:
    """Nettoyage minimal + suppression robuste du footer."""
    if not text:
        return ""

    # espaces insécables -> espaces normaux
    text = text.replace("\u00A0", " ")

    # supprimer le footer (ligne entière)
    text = re.sub(FOOTER_LINE_RE, "", text)

    # supprimer le footer s'il est "inline" (au cas où)
    text = re.sub(FOOTER_INLINE_RE, "", text)

    # normaliser les espaces
    text = re.sub(r"[ \t]+", " ", text)

    # réduire les sauts de ligne multiples
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# -----------------------------------------------------------------------------
# 2) Extraction PDF page par page
# -----------------------------------------------------------------------------
def extract_pdf_text(pdf_path: Path, max_pages: Optional[int] = None) -> str:
    """
    Extrait le texte du PDF page par page et retourne un seul grand texte.
    max_pages : utile pour tester rapidement sur un début de document.
    """
    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)

    parts: List[str] = []
    for i in tqdm(range(n_pages), desc="Extract PDF"):
        raw = reader.pages[i].extract_text() or ""
        txt = clean_text(raw)
        if txt:
            parts.append(txt)

    # Double saut de ligne entre pages pour éviter des collages bizarres
    return "\n\n".join(parts).strip()


# -----------------------------------------------------------------------------
# 3) Chunking par "Article ..."
# -----------------------------------------------------------------------------
# Exemples détectés :
#   Article L111-1
#   Article R914-13-11
#   Article D401-3
#
# La regex repère un en-tête d'article en début de ligne (ou après \n).
ARTICLE_HEADER_RE = re.compile(
    r"(?im)(^|\n)\s*Article\s+([LDR]\s*\d[\d\-]*)\b"
)


def normalize_article_id(article_raw: str) -> str:
    """Ex: 'L 111-1' -> 'L111-1'"""
    return re.sub(r"\s+", "", article_raw.strip()).upper()


def split_by_articles(full_text: str) -> List[Tuple[str, str]]:
    """
    Découpe full_text en chunks, un par article.
    Retour : liste (article_id, chunk_text).
    """
    matches = list(ARTICLE_HEADER_RE.finditer(full_text))

    if not matches:
        return [("NO_ARTICLE", full_text)]

    chunks: List[Tuple[str, str]] = []

    for i, m in enumerate(matches):
        start = m.start()
        article_id = normalize_article_id(m.group(2))

        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        chunk_text = full_text[start:end].strip()

        if len(chunk_text) < 50:
            continue

        chunks.append((article_id, chunk_text))

    return chunks


# -----------------------------------------------------------------------------
# 4) Audit : comment "s'assurer" que les articles sont complets ?
# -----------------------------------------------------------------------------
# On ne peut jamais garantir 100% avec un PDF (extraction imparfaite possible),
# MAIS on peut détecter les chunks suspects.
#
# Heuristiques (simples et utiles) :
# - le chunk ne doit contenir qu'UN seul en-tête "Article ..." (celui du début)
# - le footer ne doit plus apparaître
# - longueur minimale (sinon chunk probablement cassé)
# - fin de chunk "bizarre" (ex: finit par "de", "des", "du", etc.) => suspect
SUSPICIOUS_END_RE = re.compile(r"(?i)\b(de|des|du|d'|d’|et|ou|à|au|aux|pour|par)\s*$")

def audit_chunks(chunks: List[Dict]) -> List[Dict]:
    """Retourne une liste de chunks suspects avec raisons."""
    problems = []
    for c in chunks:
        txt = c["text"]

        # 1) plusieurs en-têtes "Article" à l'intérieur ?
        headers = list(ARTICLE_HEADER_RE.finditer(txt))
        if len(headers) > 1:
            problems.append({"article_id": c["article_id"], "reason": f"{len(headers)} en-têtes 'Article' détectés dans le même chunk"})

        # 2) footer encore présent ?
        if "Dernière modification" in txt and "Document généré" in txt and "Code de" in txt:
            problems.append({"article_id": c["article_id"], "reason": "footer encore présent (Dernière modification / Document généré)"})

        # 3) chunk trop court
        if len(txt) < 200:
            problems.append({"article_id": c["article_id"], "reason": f"chunk très court ({len(txt)} caractères)"})

        # 4) fin suspecte
        if SUSPICIOUS_END_RE.search(txt):
            problems.append({"article_id": c["article_id"], "reason": "fin de chunk suspecte (mot de liaison)"})

    return problems


# -----------------------------------------------------------------------------
# 5) Sauvegardes : JSONL + preview + audit
# -----------------------------------------------------------------------------
def save_jsonl(chunks: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_preview_md(chunks: List[Dict], out_path: Path, limit: int = 60, excerpt_chars: int = 1200) -> None:
    """
    IMPORTANT : c'est un aperçu, donc OUI : on tronque volontairement.
    Les textes complets sont dans le JSONL.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Aperçu des chunks (Article-level)\n"]
    for i, c in enumerate(chunks[:limit], 1):
        lines.append(f"## {i}. Article {c['article_id']}\n")
        excerpt = c["text"][:excerpt_chars].replace("\n", " ")
        lines.append(excerpt + "\n\n---\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_audit_md(problems: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Audit des chunks\n"]
    if not problems:
        lines.append("✅ Aucun chunk suspect détecté avec les heuristiques actuelles.\n")
    else:
        lines.append(f"⚠️ {len(problems)} signalements (un article peut apparaître plusieurs fois si plusieurs problèmes).\n")
        lines.append("\n## Détails\n")
        for p in problems:
            lines.append(f"- **{p['article_id']}** : {p['reason']}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# 6) Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Chunk PDF by 'Article ...' headers (footer-clean + audit).")
    parser.add_argument("--pdf", type=str, default="data/code_education.pdf", help="Chemin vers le PDF.")
    parser.add_argument("--out_jsonl", type=str, default="data/chunks_articles.jsonl", help="Sortie JSONL.")
    parser.add_argument("--out_preview", type=str, default="data/chunks_preview.md", help="Sortie aperçu Markdown.")
    parser.add_argument("--out_audit", type=str, default="data/chunks_audit.md", help="Sortie audit Markdown.")
    parser.add_argument("--max_pages", type=int, default=None, help="Limiter le nb de pages (debug).")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    # 1) extraction
    full_text = extract_pdf_text(pdf_path, max_pages=args.max_pages)
    if not full_text:
        raise RuntimeError("Aucun texte extrait (PDF vide ou extraction impossible).")

    # 2) split en articles
    article_chunks = split_by_articles(full_text)

    # 3) structurer sortie
    out: List[Dict] = [{"article_id": aid, "text": txt} for aid, txt in article_chunks]

    # 4) audit
    problems = audit_chunks(out)

    # 5) sauvegarder
    out_jsonl = Path(args.out_jsonl)
    out_preview = Path(args.out_preview)
    out_audit = Path(args.out_audit)

    save_jsonl(out, out_jsonl)
    save_preview_md(out, out_preview, limit=80, excerpt_chars=1200)
    save_audit_md(problems, out_audit)

    print(f"OK: chunks={len(out)}")
    print(f"- JSONL   : {out_jsonl}")
    print(f"- PREVIEW : {out_preview} (aperçu tronqué)")
    print(f"- AUDIT   : {out_audit} ({'0 problème' if not problems else str(len(problems)) + ' signalements'})")


if __name__ == "__main__":
    main()
