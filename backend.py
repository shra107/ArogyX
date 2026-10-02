"""
backend.py
----------
Core processing layer for ArogyX Healthcare Fraud Detector.

Responsibilities:
  - OCR (image → text → DataFrame)
  - PDF parsing (PDF → text → DataFrame)
  - Fraud detection via price_lookup.analyse_bill()
"""

import re
import pandas as pd
import pytesseract
from PIL import Image
import pdfplumber

from price_lookup import analyse_bill, build_medicine_db

# --------------------------------------------------
# Tesseract Path  (Windows default install)
# --------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------------------------------
# Pre-warm: build / verify medicine DB on import
# (runs once; subsequent calls are no-ops)
# --------------------------------------------------
build_medicine_db()


# --------------------------------------------------
# Utility: clean up a single item name
# --------------------------------------------------
def normalize_item(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------
# Fraud Detection  (main entry point for app.py)
# --------------------------------------------------
def check_fraud(bill_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a DataFrame with columns [item, quantity, billed_mrp].
    Returns an enriched fraud-analysis DataFrame from price_lookup.analyse_bill().

    Falls back to an empty result frame if input is empty.
    """
    if bill_df is None or bill_df.empty:
        return pd.DataFrame(columns=[
            "item", "quantity", "billed_mrp",
            "matched_name", "reference_price", "pack_size",
            "manufacturer", "composition",
            "match_score", "match_method",
            "billed_per_unit", "ref_per_unit",
            "overcharge_pct", "overcharge_severity",
            "extra_amount", "mrp_price", "expected_price", "status",
        ])

    # Sanitise item names before lookup
    bill_df = bill_df.copy()
    bill_df["item"] = bill_df["item"].apply(normalize_item)

    # Drop rows with no meaningful item name
    bill_df = bill_df[bill_df["item"].str.len() > 1].reset_index(drop=True)

    return analyse_bill(bill_df)


# --------------------------------------------------
# OCR: Image → DataFrame
# --------------------------------------------------
def ocr_to_dataframe(image_file) -> pd.DataFrame:
    """
    Runs Tesseract OCR on a PIL-compatible image file.
    Parses lines of the form:  <item name>  <qty>  <price>
    """
    try:
        img  = Image.open(image_file)
        text = pytesseract.image_to_string(img)
        return _parse_text_to_df(text)
    except Exception as exc:
        print(f"OCR error: {exc}")
        return pd.DataFrame(columns=["item", "quantity", "billed_mrp"])


# --------------------------------------------------
# PDF: PDF → DataFrame
# --------------------------------------------------
def pdf_to_dataframe(pdf_file) -> pd.DataFrame:
    """
    Extracts text from every page of a PDF and parses billing lines.
    """
    lines = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                lines.append(text)
        return _parse_text_to_df("\n".join(lines))
    except Exception as exc:
        print(f"PDF parsing error: {exc}")
        return pd.DataFrame(columns=["item", "quantity", "billed_mrp"])


# --------------------------------------------------
# Shared text → DataFrame parser
# --------------------------------------------------
def _parse_text_to_df(text: str) -> pd.DataFrame:
    """
    Heuristic parser for bill text.

    Handles two layouts:
      A) <name>  <qty>  <price>        (3+ tokens, last two numeric)
      B) <name>  <price>               (2+ tokens, last one numeric)

    Skips header-like lines and lines that are entirely numeric.
    """
    data = []
    skip_keywords = {"item", "medicine", "description", "qty", "quantity",
                     "price", "mrp", "amount", "total", "s.no", "sr", "no."}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Normalise separators
        line = re.sub(r"[,\t|]+", " ", line)
        parts = line.split()

        if len(parts) < 2:
            continue

        # Skip header rows
        first_token = parts[0].lower().rstrip(".")
        if first_token in skip_keywords:
            continue

        # Try layout A: last two tokens are qty (int) + price (float)
        if len(parts) >= 3:
            try:
                qty   = int(parts[-2])
                price = float(parts[-1])
                name  = " ".join(parts[:-2]).strip()
                if name and qty > 0 and price > 0:
                    data.append({"item": name, "quantity": qty, "billed_mrp": price})
                    continue
            except ValueError:
                pass

        # Try layout B: last token is price (float), qty defaults to 1
        try:
            price = float(parts[-1])
            name  = " ".join(parts[:-1]).strip()
            if name and price > 0:
                data.append({"item": name, "quantity": 1, "billed_mrp": price})
        except ValueError:
            pass

    return pd.DataFrame(data, columns=["item", "quantity", "billed_mrp"]) \
           if data else pd.DataFrame(columns=["item", "quantity", "billed_mrp"])
