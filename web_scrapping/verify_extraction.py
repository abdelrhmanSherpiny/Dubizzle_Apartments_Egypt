import json

# Check checkpoint
cp = json.load(open("extract_checkpoint.json", "r", encoding="utf-8"))
processed = cp.get("processed_indices", [])
print(f"Checkpoint processed count: {len(processed)}")
print(f"Last batch_start: {cp.get('last_batch_start', 'N/A')}")
print(f"Sample indices (first 10): {sorted(processed)[:10]}")
print(f"Sample indices (last 10): {sorted(processed)[-10:]}")

# Check the enriched file to see what LLM actually filled
enriched = []
with open("dubizzle_apartments_enriched.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            enriched.append(json.loads(line.strip()))

print(f"\nEnriched file has {len(enriched)} records")

# For processed indices, check what the LLM filled
FEATURES = [
    "finish_type", "compound_name", "down_payment_egp", "installment_years",
    "monthly_installment_egp", "floor_number", "delivery_date", "view_type",
    "has_garden", "has_roof", "has_elevator", "has_parking", "has_pool", "has_security",
]

# Check LLM quality on processed records
if processed:
    sample_indices = sorted(processed)[:200]
    print(f"\n=== LLM extraction quality on {len(sample_indices)} processed records ===")
    for feat in FEATURES:
        filled = sum(1 for idx in sample_indices if idx < len(enriched) and enriched[idx].get(feat))
        pct = filled / len(sample_indices) * 100
        print(f"  {feat:28s}: {filled:4d} / {len(sample_indices)} ({pct:.1f}%)")

    # Show a few examples
    print("\n=== Sample LLM extractions ===")
    count = 0
    for idx in sample_indices[:30]:
        if idx >= len(enriched):
            continue
        rec = enriched[idx]
        extracted = {f: rec.get(f) for f in FEATURES if rec.get(f)}
        if extracted:
            count += 1
            if count <= 5:
                title = (rec.get("title") or "")[:60]
                print(f"\nRecord {idx}: {title}")
                for k, v in extracted.items():
                    print(f"  {k}: {v}")
