# LLM + RAG – Code de l’éducation (Projet expérimental)

---

## 🇫🇷 Partie 1 — Description en français

### Objectif du projet
Ce projet explore la conception d’un **assistant IA local** combinant :
- un **LLM local** (via Ollama),
- un **pipeline RAG (Retrieval-Augmented Generation)**,
- appliqué au **Code de l’éducation français**.

L’objectif principal est de **comprendre, tester et structurer** une approche robuste permettant :
- d’interroger un corpus juridique complexe,
- de limiter les hallucinations des modèles de langage,
- de produire des réponses traçables et vérifiables,
- tout en restant dans un environnement **local et maîtrisé**.

Ce projet s’inscrit dans une démarche **expérimentale et pédagogique**.

---

###  État actuel du projet

#### Environnement technique
- Système : Ubuntu (WSL)
- GPU : NVIDIA RTX 4060 (8 Go)
- Environnement Python isolé : `llm_code_education_env`
- Gestion du LLM local via **Ollama**

####  Données
- Source actuelle : **Code de l’éducation au format PDF**
- Extraction du texte page par page
- Nettoyage léger du texte
- Découpage en chunks
- Vectorisation avec `sentence-transformers`
- Stockage dans un **index FAISS local**

#### Pipeline RAG (texte)
- Question utilisateur en entrée
- Recherche sémantique dans l’index FAISS
- Injection du contexte pertinent dans le LLM local
- Génération de réponses textuelles basées sur le contexte récupéré

#### Voix (expérimentation en cours)
- Intégration technique d’un moteur de **speech-to-text local**
- Validation du flux audio → texte
- Réflexion en cours sur une intégration plus ergonomique via navigateur

---

### Enseignements clés à ce stade
- Les performances d’un RAG dépendent fortement de la **qualité et de la structure de la source**.
- Un corpus juridique en PDF impose des contraintes importantes :
  - structure implicite,
  - références juridiques indirectes,
  - pagination non normative.
- Les LLM nécessitent des **garde-fous explicites** pour éviter des réponses plausibles mais incorrectes.
- La séparation claire entre :
  - récupération de l’information,
  - génération de la réponse,
  est essentielle pour améliorer la fiabilité.

---

### Prochaines étapes
- Mise en place d’un **backend FastAPI** unifié
- Capture audio côté navigateur (Web Audio API)
- Pipeline STT intégré au backend
- Renforcement du RAG avec :
  - validation explicite des sources,
  - citations construites côté code,
- Étude d’une source juridique plus structurée (XML / Légifrance)

---

### Avertissement
Ce projet est **expérimental**.  
Les réponses produites :
- ne constituent pas un avis juridique,
- peuvent être incomplètes ou inexactes,
- doivent toujours être vérifiées à partir des sources officielles.

---

## 🇬🇧 Part 2 — English description

### Project goal
This project explores the design of a **local AI assistant** combining:
- a **local LLM** (via Ollama),
- a **RAG (Retrieval-Augmented Generation) pipeline**,
- applied to the **French Code of Education**.

The main objective is to **understand, test, and structure** a robust approach to:
- query a complex legal corpus,
- reduce LLM hallucinations,
- produce traceable and verifiable answers,
- while keeping everything **local and controlled**.

This is an **experimental and educational** project.

---

### Current project status

#### Technical environment
- System: Ubuntu (WSL)
- GPU: NVIDIA RTX 4060 (8 GB)
- Isolated Python environment: `llm_code_education_env`
- Local LLM management via **Ollama**

#### Data
- Current source: **French Code of Education (PDF format)**
- Page-by-page text extraction
- Light text cleaning
- Chunking
- Embeddings with `sentence-transformers`
- Local **FAISS vector index**

#### RAG pipeline (text-based)
- User question as input
- Semantic search in FAISS
- Injection of relevant context into the local LLM
- Text-based answer generation grounded in retrieved context

#### Voice (ongoing experimentation)
- Technical validation of a **local speech-to-text** engine
- Audio → text pipeline validated
- Ongoing reflection on browser-based integration for better UX

---

### Key insights so far
- RAG performance strongly depends on **data structure and quality**.
- PDF-based legal corpora introduce significant constraints:
  - implicit structure,
  - indirect legal references,
  - non-normative pagination.
- LLMs require **explicit safeguards** to avoid plausible but incorrect answers.
- A clear separation between:
  - information retrieval,
  - answer generation,
  is critical to improve reliability.

---

### Next steps
- Unified **FastAPI backend**
- Browser-side audio capture (Web Audio API)
- Backend-integrated STT pipeline
- Stronger RAG with:
  - explicit source validation,
  - code-enforced citations,
- Evaluation of more structured legal sources (XML / Légifrance)

---

### Disclaimer
This project is **experimental**.  
The generated answers:
- do not constitute legal advice,
- may be incomplete or inaccurate,
- must always be verified against official sources.
