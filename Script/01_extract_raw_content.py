import json
from pathlib import Path
import pdfplumber

# ----------------- paths -----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "Research_Paper"
OUTPUT_DIR = PROJECT_ROOT / "Dataset" / "raw"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_one_pdf(pdf_path: Path):
    """
    Extract plain text and simple tables from a single PDF.

    INCREMENTAL:
      - If the corresponding *_raw.json already exists, we SKIP this PDF.
    """
    paper_id = pdf_path.stem
    out_path = OUTPUT_DIR / f"{paper_id}_raw.json"

    # ---- incremental behaviour: skip if already processed ----
    if out_path.exists():
        print(f"[SKIP] {pdf_path.name} -> {out_path.name} already exists")
        return

    print(f"[NEW ] Processing: {pdf_path.name}")

    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # ----- text -----
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            # ----- tables (very simple extraction) -----
            try:
                tables_raw = page.extract_tables() or []
            except Exception:
                tables_raw = []

            tables = []
            for tbl in tables_raw:
                # tbl is already a list of rows; we keep as-is
                tables.append(tbl)

            pages_data.append(
                {
                    "page_num": i,
                    "text": text,
                    "tables": tables,
                }
            )

    record = {
        "paper_id": paper_id,
        "filename": pdf_path.name,
        # store path relative to project root – useful later
        "pdf_path": pdf_path.relative_to(PROJECT_ROOT).as_posix(),
        "pages": pages_data,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"  -> written: {out_path}")


def main():
    if not PAPERS_DIR.exists():
        print(f"PDF folder not found: {PAPERS_DIR}")
        return

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in: {PAPERS_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {PAPERS_DIR}")
    for pdf_path in pdf_files:
        extract_one_pdf(pdf_path)

    print("\nDone. Raw JSON files are in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
