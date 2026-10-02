"""
price_lookup.py
---------------
Real-time medicine price lookup engine for ArogyX fraud detector.

Pipeline for each billed medicine name:
  1. Exact match against 253K Indian Medicine Dataset (SQLite)
  2. Fuzzy match via rapidfuzz (token_set_ratio, cutoff 75)
  3. RxNorm API fallback: brand → generic name, then retry steps 1-2
  4. Ingredient-level match: extract active ingredient from bill name, fuzzy against compositions

Returns structured overcharge analysis per line item.
"""

import os
import re
import sqlite3
import logging
import requests
import pandas as pd
from functools import lru_cache
from rapidfuzz import process, fuzz

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MEDICINE_CSV = os.path.join(BASE_DIR, "data", "indian_medicine_data.csv")
DB_PATH      = os.path.join(BASE_DIR, "fraud_detection.db")

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Constants
# --------------------------------------------------
FUZZY_CUTOFF          = 75    # minimum score (0-100) for a fuzzy match to be accepted
RXNORM_BASE_URL       = "https://rxnav.nlm.nih.gov/REST"
RXNORM_TIMEOUT        = 5     # seconds
# Overcharge thresholds
OVERCHARGE_LOW        = 5.0   # % — minor overcharge
OVERCHARGE_MEDIUM     = 20.0  # % — significant overcharge
OVERCHARGE_HIGH       = 50.0  # % — critical overcharge

# --------------------------------------------------
# DB Bootstrap: load CSV → SQLite once
# --------------------------------------------------
def build_medicine_db(force: bool = False) -> None:
    """
    Load indian_medicine_data.csv into the 'medicines' table in fraud_detection.db.
    Skipped on subsequent runs unless force=True.
    """
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Check if already populated
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='medicines'")
    table_exists = cur.fetchone() is not None
    if table_exists and not force:
        cur.execute("SELECT COUNT(*) FROM medicines")
        count = cur.fetchone()[0]
        if count > 0:
            logger.info("Medicine DB already loaded (%d records). Skipping rebuild.", count)
            conn.close()
            return

    logger.info("Loading Indian Medicine Dataset into SQLite …")

    df = pd.read_csv(MEDICINE_CSV, encoding="utf-8", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Rename for convenience
    df = df.rename(columns={
        "price(₹)":           "price",
        "Is_discontinued":    "is_discontinued",
        "manufacturer_name":  "manufacturer",
        "pack_size_label":    "pack_size",
        "short_composition1": "composition1",
        "short_composition2": "composition2",
    })

    # Keep only active (non-discontinued) drugs
    df["is_discontinued"] = df["is_discontinued"].str.upper().str.strip()
    df = df[df["is_discontinued"] != "TRUE"].copy()

    # Clean price — drop rows with no price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    # Normalised name column for matching
    df["name_clean"] = df["name"].apply(_clean_name)

    # Persist
    cur.execute("DROP TABLE IF EXISTS medicines")
    df.to_sql("medicines", conn, if_exists="replace", index=False)

    # Index for fast LIKE queries
    cur.execute("CREATE INDEX IF NOT EXISTS idx_name_clean ON medicines(name_clean)")
    conn.commit()
    conn.close()
    logger.info("Medicine DB built: %d active records.", len(df))


# --------------------------------------------------
# Name Normalisation Helpers
# --------------------------------------------------
_STRIP_PATTERNS = re.compile(
    r"\b(tablet|tab|cap|capsule|syrup|injection|inj|drops|cream|gel|ointment|"
    r"solution|suspension|powder|mg|ml|mcg|iu|units?|strip|bottle|pack|of|"
    r"each|per)\b",
    re.IGNORECASE,
)

def _clean_name(text: str) -> str:
    """Lowercase, remove dosage form / unit noise, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _STRIP_PATTERNS.sub(" ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# In-memory catalogue (loaded once per process)
# --------------------------------------------------
_catalogue: pd.DataFrame | None = None
_name_list: list[str]           = []

def _get_catalogue() -> tuple[pd.DataFrame, list[str]]:
    """Return cached catalogue DataFrame and normalised name list."""
    global _catalogue, _name_list
    if _catalogue is None:
        build_medicine_db()
        conn = sqlite3.connect(DB_PATH)
        _catalogue = pd.read_sql(
            "SELECT id, name, name_clean, price, manufacturer, pack_size, "
            "composition1, composition2 FROM medicines",
            conn,
        )
        conn.close()
        _name_list = _catalogue["name_clean"].tolist()
        logger.info("Catalogue loaded into memory: %d drugs.", len(_catalogue))
    return _catalogue, _name_list


# --------------------------------------------------
# RxNorm API: brand → generic name resolution
# --------------------------------------------------
@lru_cache(maxsize=512)
def rxnorm_resolve(drug_name: str) -> str | None:
    """
    Call RxNorm approximate-match API.
    Returns the best canonical drug name or None on failure / no match.
    Free, no API key needed.
    """
    try:
        # Step 1: approximate match → get RxCUI
        resp = requests.get(
            f"{RXNORM_BASE_URL}/approximateTerm.json",
            params={"term": drug_name, "maxEntries": 3},
            timeout=RXNORM_TIMEOUT,
        )
        resp.raise_for_status()
        candidates = resp.json().get("approximateGroup", {}).get("candidate", [])
        if not candidates:
            return None

        rxcui = candidates[0].get("rxcui")
        if not rxcui:
            return None

        # Step 2: get canonical name for that RxCUI
        resp2 = requests.get(
            f"{RXNORM_BASE_URL}/rxcui/{rxcui}/property.json",
            params={"propName": "RxNorm Name"},
            timeout=RXNORM_TIMEOUT,
        )
        resp2.raise_for_status()
        prop_val = (
            resp2.json()
            .get("propConceptGroup", {})
            .get("propConcept", [{}])[0]
            .get("propValue")
        )
        return prop_val  # may be None if structure differs

    except Exception as exc:
        logger.debug("RxNorm lookup failed for '%s': %s", drug_name, exc)
        return None


# --------------------------------------------------
# Core Lookup: single medicine name → match result
# --------------------------------------------------
def lookup_medicine(billed_name: str) -> dict:
    """
    Match a billed medicine name against the 253K catalogue.

    Returns a dict:
        matched_name    : str  — best matched name in catalogue
        reference_price : float — catalogue price (per pack)
        pack_size       : str
        manufacturer    : str
        composition     : str  — active ingredients
        match_score     : float (0-100)
        match_method    : str  — 'exact' | 'fuzzy' | 'rxnorm' | 'ingredient' | 'not_found'
    """
    catalogue, name_list = _get_catalogue()
    query_clean = _clean_name(billed_name)

    # ── 1. Exact match ─────────────────────────────────────────────────────
    exact = catalogue[catalogue["name_clean"] == query_clean]
    if not exact.empty:
        row = exact.iloc[0]
        return _build_result(row, 100.0, "exact")

    # ── 2. Fuzzy match ──────────────────────────────────────────────────────
    match = process.extractOne(
        query_clean,
        name_list,
        scorer=fuzz.token_set_ratio,
        score_cutoff=FUZZY_CUTOFF,
    )
    if match:
        matched_clean, score, idx = match
        row = catalogue.iloc[idx]
        return _build_result(row, score, "fuzzy")

    # ── 3. RxNorm fallback → retry fuzzy ────────────────────────────────────
    generic_name = rxnorm_resolve(billed_name)
    if generic_name:
        generic_clean = _clean_name(generic_name)
        match2 = process.extractOne(
            generic_clean,
            name_list,
            scorer=fuzz.token_set_ratio,
            score_cutoff=FUZZY_CUTOFF,
        )
        if match2:
            matched_clean, score, idx = match2
            row = catalogue.iloc[idx]
            return _build_result(row, score, "rxnorm")

    # ── 4. Ingredient-level match ────────────────────────────────────────────
    # Extract the first word(s) that look like an ingredient (skip dose/form words)
    ingredient = _extract_ingredient(billed_name)
    if ingredient:
        ing_clean = _clean_name(ingredient)
        # Search composition columns for partial match
        mask = (
            catalogue["composition1"].str.lower().str.contains(ing_clean, na=False)
            | catalogue["composition2"].str.lower().str.contains(ing_clean, na=False)
        )
        ing_matches = catalogue[mask]
        if not ing_matches.empty:
            row = ing_matches.iloc[0]
            return _build_result(row, 60.0, "ingredient")

    return {
        "matched_name":    None,
        "reference_price": None,
        "pack_size":       None,
        "manufacturer":    None,
        "composition":     None,
        "match_score":     0.0,
        "match_method":    "not_found",
    }


def _build_result(row: pd.Series, score: float, method: str) -> dict:
    comp1 = str(row.get("composition1", "") or "").strip()
    comp2 = str(row.get("composition2", "") or "").strip()
    composition = comp1
    if comp2 and comp2.lower() not in ("nan", ""):
        composition = f"{comp1} + {comp2}"
    return {
        "matched_name":    row["name"],
        "reference_price": float(row["price"]),
        "pack_size":       str(row.get("pack_size", "") or ""),
        "manufacturer":    str(row.get("manufacturer", "") or ""),
        "composition":     composition,
        "match_score":     round(score, 1),
        "match_method":    method,
    }


def _extract_ingredient(text: str) -> str:
    """
    Best-effort: pull the meaningful drug word(s) from a name like
    'Paracetamol 500mg Tab' → 'Paracetamol'
    """
    text = re.sub(r"\d+(\.\d+)?\s*(mg|ml|mcg|iu|g|%)", "", text, flags=re.IGNORECASE)
    text = _STRIP_PATTERNS.sub(" ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = [w for w in text.split() if len(w) > 3]
    return " ".join(words[:2]) if words else ""


# --------------------------------------------------
# Batch Overcharge Analysis (called from backend.py)
# --------------------------------------------------
def analyse_bill(bill_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns [item, quantity, billed_mrp],
    returns an enriched DataFrame with price comparison and fraud flags.

    Output columns:
        item, quantity, billed_mrp,
        matched_name, reference_price, pack_size, manufacturer, composition,
        match_score, match_method,
        billed_per_unit, ref_per_unit,
        overcharge_pct, overcharge_severity, status
    """
    if bill_df.empty:
        return _empty_result()

    results = []
    for _, row in bill_df.iterrows():
        item_name  = str(row.get("item", "")).strip()
        quantity   = float(row.get("quantity", 1) or 1)
        billed_mrp = float(row.get("billed_mrp", 0) or 0)

        lookup = lookup_medicine(item_name)

        ref_price   = lookup["reference_price"]   # per-pack catalogue price
        pack_size   = lookup["pack_size"]

        # Derive per-unit prices where possible
        units_in_pack = _parse_pack_units(pack_size)

        if ref_price is not None and units_in_pack > 0:
            ref_per_unit    = ref_price / units_in_pack
            billed_per_unit = billed_mrp / quantity if quantity else billed_mrp
        elif ref_price is not None:
            # No pack unit info — compare total billed vs catalogue pack price × qty
            ref_per_unit    = ref_price
            billed_per_unit = billed_mrp / quantity if quantity else billed_mrp
        else:
            ref_per_unit    = None
            billed_per_unit = billed_mrp / quantity if quantity else billed_mrp

        # Overcharge calculation
        if ref_per_unit is not None and ref_per_unit > 0:
            overcharge_pct = ((billed_per_unit - ref_per_unit) / ref_per_unit) * 100
        else:
            overcharge_pct = None

        severity = _classify_severity(overcharge_pct)
        status   = _classify_status(overcharge_pct, lookup["match_method"])

        # extra_amount keeps compatibility with existing backend.py
        extra_amount = (
            round((billed_per_unit - ref_per_unit) * quantity, 2)
            if ref_per_unit is not None else 0.0
        )

        results.append({
            # original bill columns
            "item":              item_name,
            "quantity":          quantity,
            "billed_mrp":        billed_mrp,
            # match info
            "matched_name":      lookup["matched_name"],
            "reference_price":   ref_price,
            "pack_size":         pack_size,
            "manufacturer":      lookup["manufacturer"],
            "composition":       lookup["composition"],
            "match_score":       lookup["match_score"],
            "match_method":      lookup["match_method"],
            # price analysis
            "billed_per_unit":   round(billed_per_unit, 2) if billed_per_unit else None,
            "ref_per_unit":      round(ref_per_unit, 2) if ref_per_unit else None,
            "overcharge_pct":    round(overcharge_pct, 1) if overcharge_pct is not None else None,
            "overcharge_severity": severity,
            "extra_amount":      extra_amount,
            # legacy column
            "mrp_price":         ref_per_unit,
            "expected_price":    round(ref_per_unit * quantity, 2) if ref_per_unit else None,
            "status":            status,
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _parse_pack_units(pack_size: str) -> int:
    """
    Extract the number of units from strings like:
    'strip of 10 tablets' → 10
    'bottle of 100 ml Syrup' → 0 (can't normalise volume to units)
    '1 Tablet' → 1
    """
    if not pack_size:
        return 0
    # Match patterns like "10 tablets", "strip of 30", "30 capsules"
    m = re.search(r"\b(\d+)\s*(tablet|cap|capsule|tablet|unit|vial|sachet|patch)", pack_size, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # fallback: first standalone number
    m2 = re.search(r"\b(\d+)\b", pack_size)
    if m2:
        val = int(m2.group(1))
        return val if 1 <= val <= 1000 else 0
    return 0


def _classify_severity(pct: float | None) -> str:
    if pct is None:
        return "Unknown"
    if pct <= 0:
        return "None"
    if pct < OVERCHARGE_LOW:
        return "Negligible"
    if pct < OVERCHARGE_MEDIUM:
        return "Low"
    if pct < OVERCHARGE_HIGH:
        return "Medium"
    return "Critical"


def _classify_status(pct: float | None, method: str) -> str:
    if method == "not_found":
        return "MRP Not Found"
    if pct is None:
        return "MRP Not Found"
    if pct > OVERCHARGE_LOW:
        return "Fraud Detected"
    return "Valid"


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "item", "quantity", "billed_mrp",
        "matched_name", "reference_price", "pack_size", "manufacturer", "composition",
        "match_score", "match_method",
        "billed_per_unit", "ref_per_unit",
        "overcharge_pct", "overcharge_severity",
        "extra_amount", "mrp_price", "expected_price", "status",
    ])


# --------------------------------------------------
# CLI quick-test
# --------------------------------------------------
if __name__ == "__main__":
    test_items = pd.DataFrame([
        {"item": "Augmentin 625",        "quantity": 10, "billed_mrp": 350.00},
        {"item": "Paracetamol 500mg tab","quantity": 10, "billed_mrp": 85.00},
        {"item": "Azithral 500",         "quantity": 5,  "billed_mrp": 200.00},
        {"item": "Crocin Advance",       "quantity": 15, "billed_mrp": 120.00},
        {"item": "Dolo 650",             "quantity": 10, "billed_mrp": 60.00},
        {"item": "XYZUNKNOWNDRUG",       "quantity": 1,  "billed_mrp": 999.00},
    ])

    result = analyse_bill(test_items)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(result[["item","matched_name","billed_mrp","ref_per_unit","overcharge_pct","overcharge_severity","status","match_method"]].to_string(index=False))
