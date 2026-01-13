# LLM + RAG – Code de l’éducation

---

## 🇫🇷 Partie 1 — Description du projet (Français)

### Objectif
Ce projet vise à concevoir un **assistant IA local basé sur un pipeline RAG (Retrieval-Augmented Generation)** appliqué au **Code de l’éducation français**.  
L’objectif est de fournir des réponses **fiables, traçables et vérifiables**, fondées exclusivement sur les articles du Code de l’éducation, **sans hallucination du modèle**.

Une **mise à disposition gratuite** sous forme d’application est envisagée à terme, notamment à destination des **chefs d’établissement d’EPLE**, via une plateforme comme **Hugging Face**.

---

## Architecture du projet

```
llm_code_education/
│
├── data/
│   ├── code_education.pdf
│   ├── chunks_articles.jsonl
│   ├── chunks_preview.md
│   └── chunks_audit.md
│
├── db/
│   └── faiss_code_edu_by_article/
│
├── models/
│   └── mistral.gguf
│
├── src/
│   ├── chunk_by_article.py
│   ├── build_faiss_index.py
│   ├── rag_chat_ollama.py
│   └── rag_chat_llama.py
│
├── llm_code_education_env/
├── requirements.txt
└── README.md
```

---

## Description des dossiers

### `data/`
Contient l’ensemble des **données sources et intermédiaires** :
- `code_education.pdf` : source officielle du Code de l’éducation.
- `chunks_articles.jsonl` : base principale des articles (1 chunk = 1 article).
- `chunks_preview.md` : aperçu lisible des articles découpés.
- `chunks_audit.md` : rapport de contrôle qualité.

### `db/`
Contient l’index vectoriel :
- `faiss_code_edu_by_article/` : index FAISS construit à partir des articles du Code de l’éducation.

### `models/`
- `mistral.gguf` : modèle **Mistral Instruct** quantifié au format **GGUF**, utilisé via `llama.cpp` pour assurer la compatibilité avec un déploiement Hugging Face.

### `src/`
Contient le **code applicatif principal** :
- `chunk_by_article.py`  
  → Extraction du PDF et découpage **article par article**, avec nettoyage des en-têtes/pieds de page.
- `build_faiss_index.py`  
  → Création des embeddings et construction de l’index FAISS.
- `rag_chat_ollama.py`  
  → Version initiale du chatbot RAG utilisant **Ollama + Mistral** pour les tests locaux.
- `rag_chat_llama.py`  
  → Version adaptée pour le déploiement :
    - remplacement d’Ollama par **`llama.cpp`**,
    - utilisation d’un modèle **Mistral GGUF**,
    - paramètres optimisés (contexte, batch, nombre d’articles),
    - compatibilité **Hugging Face Spaces**.

---

## Préparation au déploiement sur Hugging Face

Les étapes suivantes ont été réalisées :
- création d’une version dédiée `rag_chat_llama.py`,
- abandon d’Ollama au profit de **`llama.cpp`**, compatible Hugging Face,
- téléchargement et intégration d’un **modèle Mistral GGUF**,
- réduction et optimisation du contexte pour de meilleures performances CPU,
- préparation à une interface web légère (Gradio / FastAPI).

---

## 🇬🇧 Part 2 — Project description (English)

### Goal
This project aims to build a **local RAG-based AI assistant** applied to the **French Code of Education**.

The objective is to provide **reliable, source-grounded answers**, strictly based on legal articles, with **verbatim citations** and strong hallucination prevention.

A **free public deployment** is planned, especially for **school principals**, via **Hugging Face Spaces**.

---

## Hugging Face deployment preparation
- Creation of a dedicated `rag_chat_llama.py` version
- Replacement of Ollama with `llama.cpp`
- Download and integration of a quantized Mistral GGUF model
- Context size and retrieval strategy optimized for CPU usage

---

## Disclaimer
This is an **experimental project**.  
Generated answers are **not legal advice** and must always be verified against official legal sources.
