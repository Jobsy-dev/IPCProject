"""
00_scan_papers.py

Incremental version:
- Scans Research_Paper/ for *.pdf
- Reads existing Dataset/papers_index.csv (if present)
- Adds ONLY new PDFs that are not already in the index
- Keeps previous rows untouched
"""

import csv
from pathlib import Path

# ---- paths relative to this script ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "Research_Paper"
DATASET_DIR = PROJECT_ROOT / "Dataset"
DATASET_DIR.mkdir(exist_ok=True)

INDEX_CSV = DATASET_DIR / "papers_index.csv"


def load_existing_index():
    """Load existing papers_index.csv if it exists."""
    if not INDEX_CSV.exists():
        return [], set()

    rows = []
    ids = set()
    with INDEX_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paper_id = row.get("paper_id")
            if paper_id:
                rows.append(row)
                ids.add(paper_id)
    return rows, ids


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        return

    # --- load existing index (if any) ---
    existing_rows, existing_ids = load_existing_index()
    print(f"Existing indexed papers: {len(existing_rows)}")

    new_rows = []

    for pdf in pdf_files:
        paper_id = pdf.stem                # filename without extension
        pdf_rel = pdf.relative_to(PROJECT_ROOT).as_posix()  # relative path

        if paper_id in existing_ids:
            # Already indexed – skip
            continue

        new_rows.append(
            {
                "paper_id": paper_id,
                "pdf_path": pdf_rel,
            }
        )

    if not new_rows and existing_rows:
        print("No new PDFs to add. Index is up to date.")
        return

    all_rows = existing_rows + new_rows

    # write CSV (rewrite full file but keep old rows + new ones)
    with INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["paper_id", "pdf_path"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Total PDFs on disk: {len(pdf_files)}")
    print(f"Previously indexed  : {len(existing_rows)}")
    print(f"Newly added         : {len(new_rows)}")
    print(f"Now indexed         : {len(all_rows)}")
    print(f"CSV written to: {INDEX_CSV}")


if __name__ == "__main__":
    main()
