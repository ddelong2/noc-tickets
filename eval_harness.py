import json
from classifier import classify_ticket
from build_knowledge_base import build_collection
from collections import defaultdict


def run_evaluation(dataset_path: str, collection,
                   prompt_version: str = "v2",
                   sample_size: int = 500) -> dict:

    with open(dataset_path) as f:
        dataset = json.load(f)[:sample_size]

    results = []
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    for ticket in dataset:
        prediction = classify_ticket(
            ticket["text"], collection, prompt_version
        )

        is_correct = prediction["classification"] == ticket["true_label"]
        confusion_matrix[ticket["true_label"]][prediction["classification"]] += 1

        results.append({
            "ticket_id": ticket["ticket_id"],
            "true_label": ticket["true_label"],
            "predicted": prediction["classification"],
            "confidence": prediction["confidence"],
            "summary": prediction["summary"],
            "latency_ms": prediction["latency_ms"],
            "correct": is_correct
        })

    # Métriques
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total
    avg_latency = sum(r["latency_ms"] for r in results) / total

    # Accuracy par catégorie
    per_category = {}
    for category in set(r["true_label"] for r in results):
        cat_results = [r for r in results if r["true_label"] == category]
        cat_correct = sum(1 for r in cat_results if r["correct"])
        per_category[category] = {
            "accuracy": cat_correct / len(cat_results),
            "count": len(cat_results)
        }

    # Rapport final
    report = {
        "prompt_version": prompt_version,
        "sample_size": total,
        "overall_accuracy": round(accuracy, 4),
        "acceptance_criteria_met": accuracy >= 0.85,  # seuil défini avant le POC
        "avg_latency_ms": round(avg_latency),
        "p95_latency_ms": sorted([r["latency_ms"] for r in results])[int(total * 0.95)],
        "per_category_accuracy": per_category,
        "confusion_matrix": dict(confusion_matrix),
        "low_confidence_rate": sum(1 for r in results
                                   if r["confidence"] < 0.7) / total
    }

    return report


# Compare les deux versions de prompt
if __name__ == "__main__":
    collection = build_collection()

    # Lance l'éval sur v1 puis v2 — montre l'amélioration
    for version in ["v1", "v2"]:
        report = run_evaluation("tickets_dataset.json", collection, version)
        print(f"\n=== EVAL RESULTS — Prompt {version} ===")
        print(f"Accuracy: {report['overall_accuracy']:.1%}")
        print(f"Criteria met (>85%): {'✅ YES' if report['acceptance_criteria_met'] else '❌ NO'}")
        print(f"Avg latency: {report['avg_latency_ms']}ms")
        print(f"Low confidence rate: {report['low_confidence_rate']:.1%}")

        with open(f"eval_report_{version}.json", "w") as f:
            json.dump(report, f, indent=2)
