import json

# Load checkpoint to get processed indices
cp = json.load(open("extract_checkpoint.json"))
processed_idx = sorted(set(cp.get("processed_indices", [])))
print(f"LLM processed records: {len(processed_idx)}")

# Load original (raw) data
orig = []
with open("dubizzle_apartments_egypt.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            orig.append(json.loads(line.strip()))

# For those 180 records: run regex, then compare with what LLM would add
# We need to simulate: what does regex give vs what the enriched file has
# The enriched file has regex + LLM results merged

enriched = []
with open("dubizzle_apartments_enriched.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            enriched.append(json.loads(line.strip()))

features = [
    "finish_type", "compound_name", "down_payment_egp", "installment_years",
    "floor_number", "delivery_date", "view_type",
    "has_garden", "has_roof", "has_elevator", "has_parking", "has_pool", "has_security",
]

n = len(processed_idx)
print()
print("=== LLM vs Regex comparison on processed records ===")
print(f"{'Feature':28s} {'Orig':>6s} {'Enriched':>9s} {'LLM added':>10s}")
print("-" * 56)

for feat in features:
    orig_count = 0
    enr_count = 0
    for idx in processed_idx:
        if idx < len(orig) and idx < len(enriched):
            if orig[idx].get(feat):
                orig_count += 1
            if enriched[idx].get(feat):
                enr_count += 1
    added = enr_count - orig_count
    pct_boost = (added / max(n, 1)) * 100
    print(f"{feat:28s} {orig_count:6d} {enr_count:9d} {added:+10d} ({pct_boost:+.1f}%)")

# Also show a few specific examples where LLM found something regex missed
print()
print("=== Sample LLM extractions (where it added value) ===")
count = 0
for idx in processed_idx[:50]:
    if idx >= len(orig) or idx >= len(enriched):
        continue
    o = orig[idx]
    e = enriched[idx]
    diffs = []
    for feat in features:
        if not o.get(feat) and e.get(feat):
            diffs.append(f"  {feat}: {e[feat]}")
    if diffs:
        count += 1
        if count <= 5:
            title = (e.get("title") or "")[:60]
            print(f"\n--- Record {idx} ({title}) ---")
            for d in diffs:
                print(d)
        
print(f"\n... {count} records got LLM improvements out of {n} processed")
