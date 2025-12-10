"""
02_clean_text.py

Clean text from Dataset/raw/*_raw.json and write:

  Dataset/clean/<paper_id>_clean.txt
  Dataset/clean/<paper_id>_sentences.txt
  Dataset/clean/<paper_id>_sections.json

Incremental:
  - If all three output files already exist for a paper_id, we SKIP it.
"""

import json
import re
from pathlib import Path
import nltk

# Make sure sentence tokenizer is available
nltk.download('punkt', quiet=True)

# ----------------- paths -----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "Dataset" / "raw"
CLEAN_DIR = PROJECT_ROOT / "Dataset" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Basic cleanup: remove page numbers, extra spaces, headers."""
    if not isinstance(text, str):
        text = str(text)

    # remove ECSS / NASA headers (very rough)
    text = re.sub(r"ECSS[\s\S]{0,40}", "", text)
    text = re.sub(r"NASA[\s\S]{0,40}", "", text)

    # remove isolated page numbers on their own line
    text = re.sub(r"\n\d+\s*\n", "\n", text)

    # merge multiple spaces / newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sections(text: str):
    """Very simple rule-based section split."""
    sections = re.split(r"\n\s*(\d+(\.\d+)?[A-Za-z ]{0,40})\s*\n", text)
    return [s.strip() for s in sections if len(s.strip()) > 40]


def process_paper(raw_file: Path):
    paper_id = raw_file.stem.replace("_raw", "")
    print(f"Processing {paper_id} ...")

    # incremental outputs
    clean_txt_path = CLEAN_DIR / f"{paper_id}_clean.txt"
    sent_path = CLEAN_DIR / f"{paper_id}_sentences.txt"
    section_path = CLEAN_DIR / f"{paper_id}_sections.json"

    # If all 3 already exist, skip
    if clean_txt_path.exists() and sent_path.exists() and section_path.exists():
        print(f"[SKIP] {paper_id} – cleaned files already exist.")
        return

    with raw_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])
    text = "\n".join([p.get("text", "") for p in pages])
    cleaned = clean_text(text)

    # Save cleaned full text
    clean_txt_path.write_text(cleaned, encoding="utf-8")

    # Split into sentences
    sentences = nltk.sent_tokenize(cleaned)
    sent_path.write_text("\n".join(sentences), encoding="utf-8")

    # Store sections
    sections = split_sections(cleaned)
    section_path.write_text(json.dumps(sections, indent=2), encoding="utf-8")

    print(f" → Saved clean text + sentences + sections for: {paper_id}")


def main():
    if not RAW_DIR.exists():
        print(f"Raw directory not found: {RAW_DIR}")
        print("Run 01_extract_raw_content.py first.")
        return

    raw_files = sorted(RAW_DIR.glob("*_raw.json"))
    print(f"Found {len(raw_files)} raw JSON files.")

    for rf in raw_files:
        process_paper(rf)

    print("\nAll papers cleaned (incremental).")


if __name__ == "__main__":
    main()

