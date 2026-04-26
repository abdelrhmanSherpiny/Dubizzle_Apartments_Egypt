#!/usr/bin/env python3
"""
Phase 2 — LLM Feature Extraction from Dubizzle Apartment Descriptions.

Uses Google Gemini Flash (free via AI Studio) to extract structured features
from the description text of scraped apartment listings.

Strategy:
  1. Regex pre-pass: extract obvious patterns (finish type, floor, etc.)
  2. Gemini batch pass: process remaining gaps in batches of 10
  3. Merge results back into the JSONL + export enriched CSV

Usage:
  python extract_features.py                    # full run
  python extract_features.py --regex-only       # regex pass only (no API)
  python extract_features.py --resume           # resume from checkpoint
  python extract_features.py --sample 100       # test on 100 records
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = ""
JSONL_FILE = "dubizzle_apartments_egypt.jsonl"
ENRICHED_JSONL = "dubizzle_apartments_enriched.jsonl"
ENRICHED_CSV = "dubizzle_apartments_enriched.csv"
CHECKPOINT_FILE = "extract_checkpoint.json"
BATCH_SIZE = 10  # descriptions per Gemini request
RPM_LIMIT = 1000  # paid tier: much higher limit
FEATURES_TO_EXTRACT = [
    "finish_type",
    "compound_name",
    "down_payment_egp",
    "installment_years",
    "monthly_installment_egp",
    "floor_number",
    "delivery_date",
    "view_type",
    "has_garden",
    "has_roof",
    "has_elevator",
    "has_parking",
    "has_pool",
    "has_security",
]

# ── Regex Extraction ─────────────────────────────────────────────────────────

# Finish type patterns (English + Arabic)
FINISH_PATTERNS = [
    (r"(?:ultra\s*)?super\s*(?:lux|luxe|luxury)", "super_lux"),
    (r"fully[_\s-]*finished", "fully_finished"),
    (r"semi[_\s-]*finished", "semi_finished"),
    (r"un[_\s-]*finished|core\s*(?:and|&)\s*shell", "unfinished"),
    (r"\blux(?:ury)?\b(?!\s*finish)", "lux"),
    # Arabic
    (r"سوبر\s*لوكس", "super_lux"),
    (r"تشطيب\s*كامل|متشطب", "fully_finished"),
    (r"نصف\s*تشطيب", "semi_finished"),
    (r"بدون\s*تشطيب|على\s*الطوب", "unfinished"),
    (r"لوكس", "lux"),
]

# Floor patterns
FLOOR_PATTERNS = [
    (r"ground\s*floor", 0),
    (r"(?:(\d{1,2})(?:st|nd|rd|th)\s*floor)", None),  # "3rd floor" → capture group
    (r"floor\s*(?:no\.?\s*)?(\d{1,2})", None),
    (r"الدور\s*(?:الأرضي|الارضي)", 0),
    (r"الدور\s*(\d{1,2})", None),
]

# Delivery / ready-to-move patterns
DELIVERY_PATTERNS = [
    (r"ready\s*(?:to\s*)?move|immediate\s*(?:delivery|receipt|move)", "immediate"),
    (r"(?:delivery|استلام|تسليم)\s*(?:in\s*)?(\d{4})", None),
    (r"(\d{4})\s*(?:delivery|استلام|تسليم)", None),
    (r"استلام\s*فوري|جاهز", "immediate"),
]

# View patterns
VIEW_PATTERNS = [
    (r"lagoon\s*view|view\s*(?:on\s*)?lagoon", "lagoon"),
    (r"garden\s*view|view\s*(?:on\s*)?garden", "garden"),
    (r"pool\s*view|view\s*(?:on\s*)?(?:swimming\s*)?pool", "pool"),
    (r"landscape\s*view|view\s*(?:on\s*)?landscape", "landscape"),
    (r"(?:open|wide|panoramic)\s*view", "panoramic"),
    (r"street\s*view|view\s*(?:on\s*)?street", "street"),
    (r"lake\s*view|view\s*(?:on\s*)?lake", "lake"),
    (r"sea\s*view|view\s*(?:on\s*)?sea", "sea"),
    (r"park\s*view", "park"),
]

# Down payment patterns
DP_PATTERNS = [
    r"(?:down\s*payment|dp|مقدم)\s*(?:of\s*)?(?:EGP\s*)?([0-9,]+(?:\.\d+)?)",
    r"([0-9,]+(?:\.\d+)?)\s*(?:down\s*payment|dp|مقدم)",
    r"(?:down\s*payment|dp)\s*(\d+%)",
]

# Installment patterns
INST_PATTERNS = [
    r"(?:installment|install|تقسيط)\s*(?:up\s*to\s*|over\s*)?(\d+)\s*(?:year|yr|سنة|سنوات)",
    r"(\d+)\s*(?:year|yr|سنة|سنوات)\s*(?:installment|install|تقسيط|أقساط)",
    r"(?:monthly\s*installment)\s*(?:of\s*)?(?:EGP\s*)?([0-9,]+)",
]

# Boolean features
BOOL_PATTERNS = {
    "has_garden":   [r"\bgarden\b", r"حديقة", r"جنينة"],
    "has_roof":     [r"\broof\b", r"روف", r"penthouse"],
    "has_elevator": [r"\belevator\b|\blift\b", r"أسانسير|مصعد"],
    "has_parking":  [r"\bparking\b|\bgarage\b", r"جراج|بارك"],
    "has_pool":     [r"\bswimming\s*pool\b|\bpool\b", r"حمام\s*سباحة"],
    "has_security": [r"\bsecurity\b|\bgated\b", r"أمن|حراسة|سيكيورتي"],
}

# Known compound names to match
KNOWN_COMPOUNDS = [
    "Mountain View iCity", "Mountain View Hyde Park", "Mountain View Chillout Park",
    "Fifth Square", "Al Marasem", "Hyde Park", "Palm Hills", "Mivida",
    "Eastown", "Villette", "Lake View", "Galleria Moon Valley", "District 5",
    "Creek Town", "The Address East", "Beit Al Watan", "El Patio Oro",
    "O West", "Badya", "Palm Parks", "ZED West", "Village West",
    "Sodic Westown", "VYE Sodic", "Green Revolution", "Beverly Hills",
    "Zayed Dunes", "Karmell", "Ivoire", "Dejoya", "Solana",
    "El Khamayel", "Madinaty", "Rehab City", "New Giza", "Dreamland",
    "Pyramids Wales", "Tesla Residence", "Ashgar City", "Joulz",
    "Mountain View October", "Degla Palms", "Icon Residence",
    "Sarai", "Taj City", "SODIC East", "Il Bosco", "Katameya Heights",
    "Swan Lake", "Stone Park", "Azad", "Capital Gardens",
    "Midtown", "The Waterway", "River Green",
]


def regex_extract(text: str) -> dict:
    """Extract features from text using regex patterns."""
    if not text:
        return {}

    result = {}
    text_lower = text.lower()

    # ── Finish type ──
    for pattern, value in FINISH_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            result["finish_type"] = value
            break

    # ── Floor ──
    for pattern, fixed_val in FLOOR_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            if fixed_val is not None:
                result["floor_number"] = fixed_val
            elif m.group(1):
                try:
                    result["floor_number"] = int(m.group(1))
                except ValueError:
                    pass
            break

    # ── Delivery ──
    for pattern, fixed_val in DELIVERY_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            if fixed_val is not None:
                result["delivery_date"] = fixed_val
            elif m.group(1):
                result["delivery_date"] = m.group(1)
            break

    # ── View ──
    for pattern, value in VIEW_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            result["view_type"] = value
            break

    # ── Down payment ──
    for pattern in DP_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            if "%" in val:
                result["down_payment_pct"] = val
            else:
                try:
                    result["down_payment_egp"] = float(val)
                except ValueError:
                    pass
            break

    # ── Installments ──
    for pattern in INST_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            try:
                v = float(val)
                if "monthly" in pattern:
                    result["monthly_installment_egp"] = v
                else:
                    result["installment_years"] = int(v)
            except ValueError:
                pass
            break

    # ── Boolean features ──
    for feat, patterns in BOOL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result[feat] = True
                break

    # ── Compound name ──
    for compound in KNOWN_COMPOUNDS:
        if compound.lower() in text_lower:
            result["compound_name"] = compound
            break

    return result


# ── Gemini LLM Extraction ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a real estate data extraction expert. Given apartment listing descriptions, extract structured features.

For each listing, return a JSON object with these fields (use null if not found):
- "finish_type": one of "super_lux", "fully_finished", "semi_finished", "lux", "unfinished", or null
- "compound_name": the project/compound name (e.g. "Mountain View iCity"), or null
- "down_payment_egp": numeric down payment in EGP, or null
- "installment_years": number of years for installments, or null
- "monthly_installment_egp": monthly installment in EGP, or null  
- "floor_number": integer floor number (0=ground), or null
- "delivery_date": "immediate" or a year like "2026", or null
- "view_type": one of "garden", "pool", "landscape", "lagoon", "panoramic", "street", "lake", "sea", "park", or null
- "has_garden": true/false
- "has_roof": true/false
- "has_elevator": true/false
- "has_parking": true/false
- "has_pool": true/false
- "has_security": true/false

IMPORTANT RULES:
- Only extract what is EXPLICITLY stated. Do not infer.
- For compound_name, give the standard English name of the compound/project.
- For down_payment_egp, convert any Arabic numbers and remove commas.
- Return a JSON array with one object per listing, in the same order as input.
- Understand both English and Arabic text."""


def build_gemini_prompt(batch: list[dict]) -> str:
    """Build a prompt for a batch of records."""
    listings = []
    for i, rec in enumerate(batch):
        desc = rec.get("description", "") or ""
        title = rec.get("title", "") or ""
        text = f"{title}\n{desc}".strip()[:800]  # cap to save tokens
        listings.append(f"=== Listing {i+1} ===\n{text}")

    return (
        f"Extract features from these {len(batch)} apartment listings.\n"
        f"Return ONLY a JSON array of {len(batch)} objects.\n\n"
        + "\n\n".join(listings)
    )


def call_gemini(prompt: str, client, model_name: str) -> list[dict] | None:
    """Call Gemini and parse the JSON response, with retry on rate limit."""
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.1,
                    "max_output_tokens": 4096,
                    "response_mime_type": "application/json",
                },
            )
            text = response.text.strip()

            # Parse JSON
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                logger.warning("Unexpected response type: %s", type(data))
                return None

        except json.JSONDecodeError as e:
            logger.warning("JSON parse error: %s", e)
            return None
        except Exception as e:
            err_str = str(e)
            if any(code in err_str for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                # Extract retry delay from error message
                wait = 30 * attempt  # default backoff
                delay_match = re.search(r"retry in ([\d.]+)s", err_str, re.IGNORECASE)
                if delay_match:
                    wait = float(delay_match.group(1)) + 2
                logger.info(
                    "  ⏳ Retryable error (attempt %d/%d). Waiting %.0fs...",
                    attempt, max_retries, wait,
                )
                time.sleep(wait)
                continue
            else:
                logger.warning("Gemini API error: %s", e)
                return None

    logger.warning("  ⚠ Max retries reached for this batch.")
    return None


# ── Checkpoint ───────────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_indices": [], "last_batch_start": 0}


def save_checkpoint(path: str, cp: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cp, f)


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract features from descriptions using Regex + Gemini")
    parser.add_argument("--regex-only", action="store_true", help="Only run regex pass (no API calls)")
    parser.add_argument("--resume", action="store_true", help="Resume Gemini pass from checkpoint")
    parser.add_argument("--sample", type=int, default=0, help="Process only N records (for testing)")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_path = os.path.join(script_dir, JSONL_FILE)
    enriched_path = os.path.join(script_dir, ENRICHED_JSONL)
    csv_path = os.path.join(script_dir, ENRICHED_CSV)
    cp_path = os.path.join(script_dir, CHECKPOINT_FILE)

    # On resume, prefer the enriched file (it has LLM data from previous runs)
    load_path = jsonl_path
    if args.resume and os.path.exists(enriched_path) and os.path.exists(cp_path):
        load_path = enriched_path
        logger.info("Resuming: loading from enriched file %s...", ENRICHED_JSONL)
    else:
        logger.info("Loading records from %s...", JSONL_FILE)

    records = []
    with open(load_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.sample > 0:
        records = records[:args.sample]

    total = len(records)
    logger.info("Loaded %d records.", total)

    # ═════════════════════════════════════════════════════════════════════
    # STEP 1: Regex pass (instant, free)
    # ═════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("═══ STEP 1: Regex extraction pass ═══")

    regex_filled = {feat: 0 for feat in FEATURES_TO_EXTRACT}

    for i, rec in enumerate(records):
        text = (rec.get("title", "") or "") + " " + (rec.get("description", "") or "")
        extracted = regex_extract(text)

        for key, val in extracted.items():
            if key in FEATURES_TO_EXTRACT and not rec.get(key):
                rec[key] = val
                regex_filled[key] += 1

    logger.info("Regex pass complete. Features filled:")
    for feat, count in regex_filled.items():
        pct = count / total * 100
        if count > 0:
            logger.info("  %-28s: %6d (%4.1f%%)", feat, count, pct)

    # ── Coverage report after regex ──
    logger.info("")
    logger.info("═══ Coverage after regex pass ═══")
    for feat in FEATURES_TO_EXTRACT:
        filled = sum(1 for r in records if r.get(feat))
        logger.info("  %-28s: %6d / %d (%4.1f%%)", feat, filled, total, filled / total * 100)

    if args.regex_only:
        # Save and exit
        _save_results(records, enriched_path, csv_path)
        logger.info("Regex-only mode. Done!")
        return

    # ═════════════════════════════════════════════════════════════════════
    # STEP 2: Gemini LLM pass (for gaps)
    # ═════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("═══ STEP 2: Gemini LLM extraction pass ═══")

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    MODEL_NAME = "gemini-2.5-flash-lite"

    # Find records that still need LLM extraction
    # Priority: records missing key features that regex couldn't fill
    KEY_FEATURES = ["finish_type", "compound_name", "delivery_date"]
    needs_llm = []
    for i, rec in enumerate(records):
        missing = sum(1 for f in KEY_FEATURES if not rec.get(f))
        if missing >= 1:
            needs_llm.append(i)

    logger.info("Records needing LLM extraction: %d / %d", len(needs_llm), total)

    # Load checkpoint
    cp = load_checkpoint(cp_path) if args.resume else {"processed_indices": [], "last_batch_start": 0}
    processed = set(cp.get("processed_indices", []))

    # Filter already-processed
    remaining = [i for i in needs_llm if i not in processed]
    logger.info("Already processed: %d | Remaining: %d", len(processed), len(remaining))

    # Process in batches
    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    request_count = 0
    minute_start = time.time()
    llm_filled = {feat: 0 for feat in FEATURES_TO_EXTRACT}

    for batch_idx in range(0, len(remaining), BATCH_SIZE):
        batch_indices = remaining[batch_idx:batch_idx + BATCH_SIZE]
        batch_records = [records[i] for i in batch_indices]
        batch_num = batch_idx // BATCH_SIZE + 1

        # Rate limiting: 15 RPM
        request_count += 1
        if request_count > RPM_LIMIT:
            elapsed = time.time() - minute_start
            if elapsed < 60:
                wait = 61 - elapsed
                logger.info("  ⏳ Rate limit reached. Waiting %.0fs...", wait)
                time.sleep(wait)
            request_count = 1
            minute_start = time.time()

        logger.info(
            "  📦 Batch %d/%d (records %d-%d)...",
            batch_num, total_batches,
            batch_indices[0], batch_indices[-1],
        )

        prompt = build_gemini_prompt(batch_records)
        results = call_gemini(prompt, client, MODEL_NAME)

        if results and len(results) == len(batch_indices):
            for idx, llm_data in zip(batch_indices, results):
                for key, val in llm_data.items():
                    if key in FEATURES_TO_EXTRACT and val is not None:
                        if not records[idx].get(key):
                            records[idx][key] = val
                            llm_filled[key] += 1
                processed.add(idx)
        elif results:
            logger.warning(
                "  ⚠ Batch returned %d results for %d records. Skipping.",
                len(results), len(batch_indices),
            )
        else:
            logger.warning("  ⚠ Batch failed. Will retry on next run.")
            # Don't mark as processed so they'll be retried

        # Save checkpoint every 10 batches
        if batch_num % 10 == 0:
            cp["processed_indices"] = list(processed)
            cp["last_batch_start"] = batch_idx
            save_checkpoint(cp_path, cp)
            logger.info("  💾 Checkpoint saved (%d processed).", len(processed))

        # Save enriched data every 50 batches (so LLM results survive interrupts)
        if batch_num % 50 == 0:
            _save_results(records, enriched_path, csv_path)
            logger.info("  📊 Progress: %d/%d batches done. Data saved.", batch_num, total_batches)

    # Final checkpoint
    cp["processed_indices"] = list(processed)
    save_checkpoint(cp_path, cp)

    logger.info("")
    logger.info("LLM pass complete. Additional features filled:")
    for feat, count in llm_filled.items():
        if count > 0:
            pct = count / total * 100
            logger.info("  %-28s: %6d (%4.1f%%)", feat, count, pct)

    # ═════════════════════════════════════════════════════════════════════
    # FINAL: Save enriched data
    # ═════════════════════════════════════════════════════════════════════
    _save_results(records, enriched_path, csv_path)

    # Final coverage report
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║     FINAL FEATURE COVERAGE (Regex + LLM)                   ║")
    logger.info("╠══════════════════════════════════════════════════════════════╣")
    for feat in FEATURES_TO_EXTRACT:
        filled = sum(1 for r in records if r.get(feat))
        pct = filled / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        logger.info("║  %-24s %s %4.1f%%  ║", feat, bar, pct)
    logger.info("╚══════════════════════════════════════════════════════════════╝")

    # Clean up checkpoint
    if os.path.exists(cp_path):
        os.remove(cp_path)
        logger.info("Checkpoint cleaned up.")


def _save_results(records: list[dict], enriched_path: str, csv_path: str):
    """Save enriched records as JSONL and CSV."""
    # JSONL
    with open(enriched_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), os.path.basename(enriched_path))

    # CSV
    import csv
    if records:
        fieldnames = list(records[0].keys())
        # Ensure new features are included
        for feat in FEATURES_TO_EXTRACT:
            if feat not in fieldnames:
                fieldnames.append(feat)

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        logger.info("Saved %d records → %s", len(records), os.path.basename(csv_path))


if __name__ == "__main__":
    main()
