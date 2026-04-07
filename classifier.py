import anthropic, chromadb, json, time
from sentence_transformers import SentenceTransformer

client = anthropic.Anthropic()
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

TAXONOMY = ["fiber_cut", "power_outage", "router_misconfiguration",
            "ddos_attack", "planned_maintenance"]

# Versions de prompt — c'est ici que tu montres la discipline de versioning
PROMPT_VERSIONS = {
    "v1": """You are a NOC classification system.
Classify the following incident ticket into exactly one category: {taxonomy}.
Return JSON only: {{"classification": "category", "confidence": 0.0-1.0, "summary": "one sentence"}}

Ticket: {ticket}""",

    "v2": """You are an expert NOC engineer at a major telecom operator.

RELEVANT CONTEXT FROM KNOWLEDGE BASE:
{rag_context}

Classify this incident ticket into exactly one of these categories: {taxonomy}
Rules:
- Choose the PRIMARY cause, not secondary effects
- If ambiguous between two categories, pick the most operationally actionable one
- Confidence < 0.7 means you're uncertain

Return JSON only:
{{
  "classification": "category",
  "confidence": 0.0-1.0,
  "summary": "one sentence max 20 words, action-oriented",
  "recommended_runbook": "runbook id if applicable"
}}

Ticket: {ticket}"""
}

def retrieve_context(ticket_text: str, collection, n_results: int = 2) -> str:
    """RAG: récupère les documents les plus pertinents de la KB"""
    query_embedding = embed_model.encode([ticket_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    context_parts = []
    for doc in results['documents'][0]:
        context_parts.append(f"- {doc[:200]}...")
    return "\n".join(context_parts)

def classify_ticket(ticket_text: str, collection,
                    prompt_version: str = "v2") -> dict:
    start = time.time()

    # RAG retrieval
    rag_context = retrieve_context(ticket_text, collection)

    # Construction du prompt
    prompt = PROMPT_VERSIONS[prompt_version].format(
        taxonomy=", ".join(TAXONOMY),
        rag_context=rag_context,
        ticket=ticket_text
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Haiku pour la latence/coût sur du volume
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    latency_ms = (time.time() - start) * 1000
    result = json.loads(response.content[0].text)
    result["latency_ms"] = round(latency_ms)
    result["prompt_version"] = prompt_version

    return result
