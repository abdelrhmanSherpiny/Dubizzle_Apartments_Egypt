import json
from collections import Counter

records = []
with open("dubizzle_apartments_egypt.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

total = len(records)
w_desc = sum(1 for r in records if r.get("description"))
avg_desc_len = sum(len(r.get("description", "")) for r in records) / max(w_desc, 1)

cols = set()
for r in records[:100]:
    cols.update(r.keys())

print(f"Total records:        {total:,}")
print(f"With description:     {w_desc:,} ({w_desc/total*100:.0f}%)")
print(f"Avg description len:  {avg_desc_len:.0f} chars")
print()
print(f"Existing columns ({len(cols)}):")
for c in sorted(cols):
    print(f"  - {c}")
print()

# Show 3 sample descriptions
print("=== SAMPLE DESCRIPTIONS ===")
samples = [r for r in records[100:200] if r.get("description")][:3]
for i, r in enumerate(samples):
    desc = r.get("description", "")[:600]
    city = r.get("city", "?")
    price = r.get("price", "?")
    print(f"\n--- Sample {i+1} (city: {city}, price: {price}) ---")
    print(desc)
    print("...")
