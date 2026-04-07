import anthropic, json, random

client = anthropic.Anthropic()

TAXONOMY = [
    "fiber_cut",
    "power_outage",
    "router_misconfiguration",
    "ddos_attack",
    "planned_maintenance"
]

def generate_ticket(category: str, ticket_id: int) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""Generate a realistic NOC (Network Operations Center) incident ticket
            for category: {category}.
            Include: timestamp, severity (P1-P4), affected equipment, symptoms,
            initial diagnosis. 2-3 sentences. Realistic telecom jargon.
            Return JSON only: {{"ticket_id": {ticket_id}, "text": "...", "true_label": "{category}"}}"""
        }]
    )
    return json.loads(response.content[0].text)

# Génère 500 tickets (100 par catégorie)
dataset = []
for category in TAXONOMY:
    for i in range(100):
        ticket = generate_ticket(category, len(dataset))
        dataset.append(ticket)

with open("tickets_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
