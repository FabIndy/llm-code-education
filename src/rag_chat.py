import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


DB_DIR = "db/faiss_code_edu_pdf"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"  

SYSTEM = """Tu es un assistant juridique spécialisé sur le Code de l'éducation français.

Règles STRICTES (non négociables) :
1) Tu réponds UNIQUEMENT avec les informations présentes dans CONTEXTE.
2) Si le CONTEXTE ne parle pas explicitement de l’objet demandé (ex: EPLE vs EPSCP), tu DOIS répondre :
   "Je ne peux pas répondre avec certitude à partir des extraits disponibles."
3) Tu ne complètes pas avec des connaissances générales.
4) Tu ajoutes des sources : à la fin de chaque phrase importante, ajoute la reference exacte de l'article.
5) Si tu détectes une confusion de sigles ou un hors-sujet (ex: EPSCP alors que question sur EPLE), tu le dis et tu refuses.
"""


def ollama_generate(prompt: str) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"]

def main():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

    print("RAG prêt. Pose ta question. (Entrée vide = quitter)")
    while True:
        q = input("\n> ").strip()
        if not q:
            break


        docs = vs.max_marginal_relevance_search(q, k=6, fetch_k=30, lambda_mult=0.7)

        context = "\n\n---\n\n".join(d.page_content for d in docs)

        prompt = f"""{SYSTEM}

CONTEXTE (extraits du Code de l'éducation - PDF):
{context}

QUESTION:
{q}

REPONSE:
"""
        try:
            ans = ollama_generate(prompt)
            print("\n" + ans)
        except Exception as e:
            print(f"\nERREUR appel Ollama: {e}")
            print("Conseil: vérifie que Ollama tourne et que http://localhost:11434 répond.")
            break

if __name__ == "__main__":
    main()
