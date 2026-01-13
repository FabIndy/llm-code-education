#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rag_chat_llama.py

MVP "juridique strict" Code de l'éducation.

ROUTAGE AUTO :
- FULLTEXT : demande "texte exact / intégral / article X" => impression exacte depuis JSONL (SANS LLM)
- LIST     : demande "quels articles parlent ..." => liste articles + extrait (SANS LLM)
- QA       : RAG => LLM (llama/Mistral) + prompt strict + VALIDATION (anti-hallucinations)

Prérequis :
- data/chunks_articles.jsonl (article-level)
- db/faiss_code_edu_by_article (FAISS)
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama



# -------------------- CONFIG --------------------
CHUNKS_PATH = Path("data/chunks_articles.jsonl")
DB_DIR = Path("db/faiss_code_edu_by_article")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "mistral:latest"

TOP_K_FETCH = 30            # nb de docs candidats récupérés
TOP_K_FINAL = 4            # nb max envoyés au LLM
SCORE_THRESHOLD = 1.10      # à ajuster (voir affichage des scores)
MAX_CHARS_PER_DOC = 800
SNIPPET_CHARS = 260

# Déclencheurs FULLTEXT
FULLTEXT_TRIGGERS = [
    "contenu exact", "texte exact", "texte intégral", "texte integral",
    "intégral", "integral", "cite intégralement", "cite integralement",
    "donne l'intégralité", "donne l'integralite", "recopie", "reproduis",
    "affiche l'article", "donne l'article", "donne moi l'article",
]

# Déclencheurs LIST
LIST_TRIGGERS = [
    "quels articles", "quelles dispositions", "articles parlent",
    "articles qui parlent", "articles sur", "donne les articles",
    "cite les articles", "références", "references",
]

# Regex article id
ARTICLE_ID_RE = re.compile(
    r"\b(?:article\s+)?([LDR]\s?\d{1,4}(?:[.-]\d+){0,4})\b",
    flags=re.IGNORECASE
)

EPLE_RE = re.compile(r"\bEPLE\b", flags=re.IGNORECASE)

# Pour valider les sorties "Articles cités : ..."
ARTICLES_CITES_RE = re.compile(r"Articles cités\s*:\s*(.*)$", flags=re.IGNORECASE | re.MULTILINE)

# initialisation llm

llm = Llama(
    model_path="models/mistral.gguf",  # Mistral GGUF
    n_ctx=2048,
    n_threads=10,
    n_batch=128,
    verbose=False,
)



# -------------------- UTILS --------------------


def llm_generate(prompt: str) -> str:
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,
    )
    return out["choices"][0]["message"]["content"].strip()



def normalize_article_id(raw: str) -> str:
    s = raw.strip().upper().replace(" ", "")
    s = s.replace(".", "-")
    return s


def extract_article_id(q: str) -> Optional[str]:
    m = ARTICLE_ID_RE.search(q)
    if not m:
        return None
    return normalize_article_id(m.group(1))


def is_fulltext_request(q: str) -> bool:
    ql = q.lower()
    if any(t in ql for t in FULLTEXT_TRIGGERS):
        return True
    aid = extract_article_id(q)
    if aid and len(ql) <= 25:
        return True
    return False


def is_list_request(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in LIST_TRIGGERS)


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def safe_snippet(text: str, n: int) -> str:
    t = " ".join((text or "").split())
    if len(t) <= n:
        return t
    return t[:n].rstrip() + "…"


def load_article_text(article_id: str) -> Optional[str]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Fichier chunks introuvable : {CHUNKS_PATH}")

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            aid = normalize_article_id(obj.get("article_id", ""))
            if aid == article_id:
                return (obj.get("text") or "").strip()
    return None


def load_vectorstore() -> FAISS:
    if not DB_DIR.exists():
        raise FileNotFoundError(f"Index FAISS introuvable : {DB_DIR}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(str(DB_DIR), embeddings, allow_dangerous_deserialization=True)


def retrieve_scored(vs: FAISS, query: str) -> List[Tuple[object, float]]:
    """
    Retourne liste (Document, score). Plus le score est PETIT, plus c'est proche (distance).
    """
    return vs.similarity_search_with_score(query, k=TOP_K_FETCH)


def filter_docs(scored: List[Tuple[object, float]]) -> List[Tuple[object, float]]:
    """
    Filtre simple par seuil + garde TOP_K_FINAL.
    Affiche les scores pour que tu ajustes SCORE_THRESHOLD.
    """
    kept = [(d, s) for (d, s) in scored if s <= SCORE_THRESHOLD]
    if not kept:
        # fallback : au moins TOP_K_FINAL meilleurs, sinon tu refuses trop souvent
        kept = sorted(scored, key=lambda x: x[1])[:TOP_K_FINAL]
    else:
        kept = sorted(kept, key=lambda x: x[1])[:TOP_K_FINAL]
    return kept


def build_context(scored_docs: List[Tuple[object, float]]) -> Tuple[str, List[str], Dict[str, str], Dict[str, float]]:
    used = []
    by_id: Dict[str, str] = {}
    by_score: Dict[str, float] = {}

    blocks = []
    for d, s in scored_docs:
        aid = d.metadata.get("article_id", "UNKNOWN")
        aid_norm = normalize_article_id(aid)
        used.append(aid_norm)

        txt = (d.page_content or "").strip()
        by_id[aid_norm] = txt
        by_score[aid_norm] = float(s)

        if len(txt) > MAX_CHARS_PER_DOC:
            txt = txt[:MAX_CHARS_PER_DOC].rstrip() + "\n[.]"

        blocks.append(f"[{aid_norm}]\n{txt}")

    used = dedupe_keep_order(used)
    return "\n\n".join(blocks), used, by_id, by_score


def eple_context_ok(question: str, by_id: Dict[str, str]) -> bool:
    """
    Si la question contient "EPLE", on veut que le contexte contienne explicitement
    des indices "collège/lycée/établissement public local d'enseignement".
    """
    if not EPLE_RE.search(question):
        return True

    joined = "\n".join(by_id.values()).lower()
    signals = [
        "établissement public local d'enseignement",
        "etablissement public local d'enseignement",
        "collège", "college", "lycée", "lycee",
        "chef d'établissement", "chef d'etablissement",
    ]
    return any(sig in joined for sig in signals)


def extract_cited_articles(answer: str) -> List[str]:
    m = ARTICLES_CITES_RE.search(answer)
    if not m:
        return []
    tail = m.group(1).strip()
    if not tail:
        return []
    parts = re.split(r"[,\s]+", tail)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # tolère "D422-15." ou "[D422-15]"
        p = p.strip("[]().;:")
        if ARTICLE_ID_RE.match(p) or re.match(r"^[LDR]\d", p, flags=re.I):
            out.append(normalize_article_id(p))
    return dedupe_keep_order(out)


def validate_answer(answer: str, allowed_articles: List[str]) -> bool:
    cited = extract_cited_articles(answer)
    allowed_set = set(allowed_articles)

    # si le LLM ne cite rien => on refuse (sinon il peut raconter)
    if not cited:
        return False

    # interdit de citer un article non présent dans la liste autorisée
    if any(c not in allowed_set for c in cited):
        return False

    return True


def build_prompt(question: str, context: str, allowed_articles: List[str]) -> str:
    allowed = ", ".join(allowed_articles)

    return f"""Tu es un assistant juridique spécialisé dans le Code de l'éducation (France).

RÈGLES ABSOLUES (non négociables) :
1) Tu réponds UNIQUEMENT à partir du CONTEXTE fourni ci-dessous.
2) Tu n'inventes rien, tu ne complètes pas, tu ne "supposes" pas. Interdiction d'utiliser :
   "on peut supposer", "il est possible que", "on peut déduire", "probablement", etc.
3) Si le CONTEXTE ne permet pas de répondre, tu dis exactement :
   "Je ne peux pas répondre avec certitude à partir des articles fournis."
4) Tu DOIS citer uniquement des articles présents dans la liste autorisée :
   {allowed}
5) Attention au sigle EPLE :
   - EPLE = établissement public local d'enseignement (collèges/lycées).
   - Ne confonds pas avec d'autres établissements.
   Si le CONTEXTE ne traite pas clairement des EPLE au sens collèges/lycées, tu refuses de conclure.

QUESTION :
{question}

CONTEXTE :
{context}

FORMAT DE SORTIE OBLIGATOIRE :
- Une réponse courte et factuelle.
- Dernière ligne STRICTE : "Articles cités : A, B, C" (uniquement parmi la liste autorisée).
"""


# -------------------- MAIN --------------------
def main():
    print("Chargement index FAISS.")
    vs = load_vectorstore()
    

    print("\nRAG prêt (llama + Mistral) — Entrée vide = quitter.\n")

    while True:
        q = input("> ").strip()
        if not q:
            break

        # --- FULLTEXT ---
        aid = extract_article_id(q)
        if aid and is_fulltext_request(q):
            txt = load_article_text(aid)
            if not txt:
                print(f"\nJe ne trouve pas l'article {aid} dans {CHUNKS_PATH}.\n")
                continue
            print(f"\n=== Article {aid} (texte exact depuis la base) ===\n")
            print(txt)
            print("\n=== Fin ===\n")
            continue

        # --- RETRIEVE (scored) ---
        scored = retrieve_scored(vs, q)
        scored = filter_docs(scored)
        context, articles, by_id, by_score = build_context(scored)

        # Affichage debug : scores
        print("\nArticles récupérés :", ", ".join(articles))
        print("Scores (plus petit = plus pertinent) :")
        for a in articles:
            print(f"- {a}: {by_score.get(a):.4f}")

        # --- LIST ---
        if is_list_request(q):
            print("\nListe (extrait) :\n")
            for a in articles:
                snippet = safe_snippet(by_id.get(a, ""), SNIPPET_CHARS)
                print(f"- Article {a} : {snippet if snippet else '(texte vide dans le chunk)'}")
            print("\nAstuce : demande ensuite 'Donne l'intégralité de l'article XXXX-YY' pour citer le texte exact.\n")
            continue

        # --- EPLE safety gate ---
        if not eple_context_ok(q, by_id):
            print("\nJe ne peux pas répondre avec certitude à partir des articles fournis.\n")
            continue

        # --- QA (LLM) ---
        prompt = build_prompt(q, context, articles)
        answer = llm_generate(prompt)

        

        # --- VALIDATION ---
        if not validate_answer(answer, articles):
            print("\nJe ne peux pas répondre avec certitude à partir des articles fournis.\n")
            continue

        print("\n" + answer + "\n")


if __name__ == "__main__":
    main()
