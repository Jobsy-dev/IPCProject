import json
from pathlib import Path

import pandas as pd


# ----------------- paths -----------------
ROOT = Path(__file__).resolve().parents[1]  #  .. (project root)
RAW_DIR = ROOT / "Dataset" / "raw"
TABLE_DIR = ROOT / "Dataset" / "tables"

INDEX_CSV = ROOT / "Dataset" / "tables_index.csv"
INDEX_XLSX = ROOT / "Dataset" / "tables_index.xlsx"


def is_probably_real_table(df: pd.DataFrame) -> bool:
    """
    Heuristic filter to drop 'fake tables' that are really just full text
    accidentally captured as tables.
    """
    if df.empty:
        return False

    # header text (first row)
    header = " ".join(str(x) for x in df.iloc[0].tolist())
    word_count = len(str(header).split())

    # Require at least 2 columns and not a super-long header of words
    if df.shape[1] >= 2 and word_count <= 30:
        return True
    if df.shape[1] >= 3 and word_count <= 50:
        return True

    # Fallback: if many numeric cells, keep it
    numeric_cells = 0
    total_cells = 0
    for val in df.stack():
        total_cells += 1
        if any(ch.isdigit() for ch in str(val)):
            numeric_cells += 1

    if total_cells > 0 and numeric_cells / total_cells > 0.5:
        return True

    # otherwise, likely a false positive
    return False


def process_raw_file(raw_path: Path, index_rows: list[dict]):
    with raw_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    paper_id = data.get("paper_id", raw_path.stem.replace("_raw", ""))
    pages = data.get("pages", [])

    print(f"  Processing {paper_id} ...")

    for page in pages:
        page_num = page.get("page_num")
        tables = page.get("tables") or []

        for t_idx, table in enumerate(tables):
            if not table:
                continue

            # Convert to DataFrame
            df_table = pd.DataFrame(table)

            # Clean: replace empty strings with NaN, drop fully empty rows
            df_table = df_table.replace(r"^\s*$", pd.NA, regex=True)
            df_table = df_table.dropna(how="all")

            if df_table.empty:
                continue

            # Filter out obviously bad tables
            if not is_probably_real_table(df_table):
                continue

            # Build file name and save CSV
            csv_name = f"{paper_id}_p{page_num}_t{t_idx + 1}.csv"
            csv_path = TABLE_DIR / csv_name

            df_table.to_csv(csv_path, index=False, encoding="utf-8")

            # For the index: short sample of header row
            sample_header = " | ".join(str(x) for x in df_table.iloc[0].tolist())
            sample_header = sample_header[:200]  # truncate

            index_rows.append(
                {
                    "paper_id": paper_id,
                    "page_num": page_num,
                    "table_idx": t_idx + 1,
                    "n_rows": df_table.shape[0],
                    "n_cols": df_table.shape[1],
                    "csv_path": str(csv_path.relative_to(ROOT)),
                    "sample_header": sample_header,
                }
            )


def main():
    print(f"RAW_DIR   = {RAW_DIR}")
    print(f"TABLE_DIR = {TABLE_DIR}")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        print("Raw JSON folder not found. Run 01_extract_raw_content.py first.")
        return

    raw_files = sorted(RAW_DIR.glob("*_raw.json"))
    if not raw_files:
        print("No *_raw.json files found. Run 01_extract_raw_content.py first.")
        return

    print(f"Found {len(raw_files)} raw JSON file(s).")

    # -------- incremental: load existing index, if any --------
    if INDEX_CSV.exists():
        df_existing = pd.read_csv(INDEX_CSV, encoding="utf-8")
        index_rows = df_existing.to_dict(orient="records")
        processed_papers = set(df_existing["paper_id"].unique())
        print(
            f"Loaded {len(df_existing)} existing tables "
            f"for {len(processed_papers)} paper(s)."
        )
    else:
        index_rows = []
        processed_papers = set()
        print("No tables_index.csv found – starting new index.")

    new_tables_count = 0

    # -------- process only new papers (not in processed_papers) --------
    for raw_path in raw_files:
        paper_id = raw_path.stem.replace("_raw", "")

        if paper_id in processed_papers:
            print(f"[SKIP] {paper_id} – tables already indexed.")
            continue

        before = len(index_rows)
        process_raw_file(raw_path, index_rows)
        after = len(index_rows)
        added = after - before
        new_tables_count += added
        processed_papers.add(paper_id)
        print(f"  -> {added} table(s) added for {paper_id}")

    if not index_rows:
        print("No tables extracted.")
        return

    # -------- write updated index (existing + new) --------
    df_index = pd.DataFrame(index_rows)
    df_index.to_csv(INDEX_CSV, index=False, encoding="utf-8")
    df_index.to_excel(INDEX_XLSX, index=False)

    print(f"\nNew tables extracted this run: {new_tables_count}")
    print(f"Total tables in index now   : {len(df_index)}")
    print(f"Index CSV written to   : {INDEX_CSV}")
    print(f"Index Excel written to : {INDEX_XLSX}")
    print(f"Individual tables in   : {TABLE_DIR}")


if __name__ == "__main__":
    main()
