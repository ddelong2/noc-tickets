# NOC Ticket Classification System

RAG-based classifier for NOC incident tickets with prompt versioning and eval harness.

## Installation

```bash
pip install anthropic chromadb sentence-transformers
```

## Usage

```bash
# Lance Claude Code
claude

# Tu lui dis ensuite :
# "Build me a NOC ticket classification system with RAG,
#  eval harness, and prompt versioning.
#  Start with generating 500 synthetic tickets across 5 categories"
```

## Pipeline

1. **`generate_tickets.py`** — Génère 500 tickets synthétiques (100 par catégorie)
2. **`build_knowledge_base.py`** — Indexe runbooks et incidents résolus dans ChromaDB
3. **`classifier.py`** — Classifie les tickets via RAG + Claude (prompt v1/v2)
4. **`eval_harness.py`** — Évalue accuracy, latence et métriques par catégorie
