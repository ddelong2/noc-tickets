import chromadb
from sentence_transformers import SentenceTransformer
import json

# Runbooks fictifs mais réalistes
RUNBOOKS = [
    {
        "id": "rb_fiber_cut",
        "category": "fiber_cut",
        "content": """Fiber cut resolution playbook:
        1. Identify affected segment via OTDR measurement
        2. Dispatch field team within 2h for P1, 4h for P2
        3. Activate backup path if available (MPLS FRR)
        4. Typical resolution time: 4-8h
        5. Escalate to regional NOC if >3 segments affected"""
    },
    {
        "id": "rb_ddos",
        "category": "ddos_attack",
        "content": """DDoS mitigation playbook:
        1. Confirm attack via traffic analysis (NetFlow)
        2. Activate scrubbing center if traffic >10Gbps
        3. Apply upstream BGP blackhole if customer requests
        4. Loop in Security SOC team immediately
        5. Document attack vector for post-mortem"""
    },
    # ... autres runbooks
]

# Historique d'incidents résolus (résumés)
RESOLVED_INCIDENTS = [
    {
        "id": "inc_001",
        "category": "router_misconfiguration",
        "content": ("BGP session drop on PE-router Lyon-3 after IOS upgrade. "
                    "Root cause: missing 'no auto-summary' command. "
                    "Resolution: rolled back config, re-applied with fix. Duration: 47min.")
    },
    # ...
]

def build_collection():
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("noc_knowledge_base")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    all_docs = RUNBOOKS + RESOLVED_INCIDENTS
    embeddings = model.encode([d["content"] for d in all_docs]).tolist()

    collection.add(
        documents=[d["content"] for d in all_docs],
        embeddings=embeddings,
        ids=[d["id"] for d in all_docs],
        metadatas=[{"category": d["category"]} for d in all_docs]
    )

    print(f"Knowledge base built: {len(all_docs)} documents indexed")
    return collection


if __name__ == "__main__":
    build_collection()
