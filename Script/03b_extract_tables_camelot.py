"""
03b_extract_tables_camelot.py  (INCREMENTAL VERSION)

Automatically detect which PDFs need Camelot table extraction.
Only runs Camelot on NEW or MISSING PDFs — never re-processes old ones.

• Reads all PDFs from Research_Paper/
• Loads existing tables_index.csv (if any)
• Detects which papers:
      - have NO tables extracted yet
      - OR have NO Camelot tables extracted yet
• Runs Camelot ONLY on those papers
• Appends new tables to tables_index.csv
"""

import camelot
import pandas as pd
from pathlib import Path
import json


# ----------------- Paths -----------------
ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "Research_Paper"
RAW_DIR = ROOT / "Dataset" / "raw"
TABLE_DIR = ROOT / "Dataset" / "tables"

INDEX_CSV = ROOT / "Dataset" / "tables_index.csv"
INDEX_XLSX = ROOT / "Dataset" / "tables_index.xlsx"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ----------- Relaxed table filter -----------
def is_probably_real_table(df: pd.DataFrame) -> bool:
    """Relaxed filter for Camelot tables."""
    if df.empty:
        return False

    header = " ".join(str(x) for x in df.iloc[0].tolist())
    word_count = len(header.split())

    if df.shape[1] >= 2 and word_count <= 80:
        return True

    numeric_cells = sum(any(ch.isdigit() for ch in str(v)) for v in df.stack())
    total_cells = len(df.stack())

    if total_cells > 0 and numeric_cells / total_cells > 0.25:
        return True

    return False


# ----------- Detect which papers need Camelot -----------
def detect_papers_needing_camelot(existing_df: pd.DataFrame):
    """
    Returns list of paper_ids that require Camelot extraction:
      • Raw file exists
      • But NO Camelot tables were extracted earlier
    """
    raw_papers = {p.stem.replace("_raw", "") for p in RAW_DIR.glob("*_raw.json")}

    if existing_df.empty:
        existing_papers = set()
    else:
        existing_papers = set(existing_df["paper_id"].unique())

    # Papers that have *no* tables at all → must run Camelot
    missing_all = raw_papers - existing_papers

    # Papers that have tables but NOT from Camelot (file contains no "*camelot*.csv")
    papers_missing_camelot = set()
    for pid in existing_papers:
        rows = existing_df[existing_df["paper_id"] == pid]
        paths = rows["csv_path"].astype(str).tolist()

        has_camelot = any("camelot" in p.lower() for p in paths)
        if not has_camelot:
            papers_missing_camelot.add(pid)

    # Final union
    need_camelot = missing_all | papers_missing_camelot

    print(f"\n→ Papers needing Camelot: {need_camelot}\n")
    return list(need_camelot)


# ----------- Camelot extraction -----------
def camelot_extract_for_paper(paper_id: str, index_rows: list, existing_keys: set):
    pdf_path = PDF_DIR / f"{paper_id}.pdf"
    if not pdf_path.exists():
        print(f"[SKIP] PDF not found: {pdf_path.name}")
        return

    print(f"\n[CAMEL0T] Extracting tables from: {pdf_path.name}")

    # Try lattice
    try:
        tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
    except:
        tables = []

    # Try stream
    if len(tables) == 0:
        try:
            print("  Lattice failed → trying stream…")
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
        except:
            tables = []

    if len(tables) == 0:
        print("  No Camelot tables found.")
        return

    print(f"  Camelot detected {len(tables)} tables")

    kept = 0
    for idx, t in enumerate(tables, start=1):
        df = t.df
        page_num = t.page

        df = df.replace(r"^\s*$", pd.NA, regex=True)
        df = df.dropna(how="all")

        if df.empty:
            continue

        if not is_probably_real_table(df):
            continue

        # Unique key
        key = (paper_id, page_num, idx)
        if key in existing_keys:
            continue

        csv_name = f"{paper_id}_p{page_num}_t{idx}_camelot.csv"
        csv_path = TABLE_DIR / csv_name

        df.to_csv(csv_path, index=False, encoding="utf-8")

        sample_header = " | ".join(str(x) for x in df.iloc[0].tolist())[:200]

        index_rows.append(
            {
                "paper_id": paper_id,
                "page_num": page_num,
                "table_idx": idx,
                "n_rows": df.shape[0],
                "n_cols": df.shape[1],
                "csv_path": str(csv_path.relative_to(ROOT)),
                "sample_header": sample_header,
            }
        )
        existing_keys.add(key)
        kept += 1

    print(f"  → Saved {kept} Camelot tables for {paper_id}")


def main():
    print("=== Incremental Camelot Table Extraction ===")

    # Load existing index
    if INDEX_CSV.exists():
        df_index = pd.read_csv(INDEX_CSV)
        index_rows = df_index.to_dict(orient="records")
        existing_keys = {(r["paper_id"], r["page_num"], r["table_idx"]) for r in index_rows}
        print(f"Loaded {len(df_index)} existing table entries.")
    else:
        df_index = pd.DataFrame()
        index_rows = []
        existing_keys = set()
        print("No existing tables_index.csv → starting fresh.")

    # Determine which papers require Camelot
    papers_to_process = detect_papers_needing_camelot(df_index)

    if not papers_to_process:
        print("All papers already have Camelot tables. Nothing to do.")
        return

    # Run Camelot only for those papers
    for pid in papers_to_process:
        camelot_extract_for_paper(pid, index_rows, existing_keys)

    # Save updated index
    df_final = pd.DataFrame(index_rows)
    df_final = df_final.sort_values(
        by=["paper_id", "page_num", "table_idx"], ignore_index=True
    )

    df_final.to_csv(INDEX_CSV, index=False)
    df_final.to_excel(INDEX_XLSX, index=False)

    print("\n=== Updated table index saved ===")
    print(f"CSV  -> {INDEX_CSV}")
    print(f"XLSX -> {INDEX_XLSX}")
    print(f"Tables saved in {TABLE_DIR}")


if __name__ == "__main__":
    main()
