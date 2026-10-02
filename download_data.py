"""
download_data.py
----------------
Run this ONCE before starting the app for the first time.
Downloads the Indian Medicine Dataset (246,064 drugs, ~30 MB)
into data/indian_medicine_data.csv, then builds the SQLite index.

Usage:
    python download_data.py
"""

import os
import sys
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "indian_medicine_data.csv")
CSV_URL  = (
    "https://raw.githubusercontent.com/junioralive/"
    "Indian-Medicine-Dataset/main/DATA/indian_medicine_data.csv"
)

def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH):
        size_mb = os.path.getsize(CSV_PATH) / 1_048_576
        print(f"Dataset already present ({size_mb:.1f} MB) — skipping download.")
        return

    print("Downloading Indian Medicine Dataset (~30 MB) ...")
    print(f"Source : {CSV_URL}")
    print(f"Target : {CSV_PATH}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "#" * int(pct // 2)
            sys.stdout.write(f"\r  [{bar:<50}] {pct:.1f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(CSV_URL, CSV_PATH, reporthook=_progress)
        print()  # newline after progress bar
        size_mb = os.path.getsize(CSV_PATH) / 1_048_576
        print(f"Download complete — {size_mb:.1f} MB saved.")
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print("Please download manually from:")
        print(f"  {CSV_URL}")
        print(f"and save to: {CSV_PATH}")
        sys.exit(1)


def build_db():
    print("\nBuilding medicine database in SQLite ...")
    try:
        from price_lookup import build_medicine_db
        build_medicine_db(force=True)
        print("Database built successfully.")
    except Exception as exc:
        print(f"DB build failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    download()
    build_db()
    print("\nSetup complete. Run the app with:")
    print("  streamlit run app.py")
