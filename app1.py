import streamlit as st
import pandas as pd
from pathlib import Path
import subprocess
import sys
import json
import shutil


# --------------------------------------------------
# Helper: robustly find project ROOT (IPC_TRAIL 7)
# --------------------------------------------------
def find_project_root(start: Path) -> Path:
    """
    Walk upwards from 'start' until we find a folder
    that has both 'Dataset' and 'Research_Paper' inside.
    That folder is treated as the project root.
    """
    for p in [start] + list(start.parents):
        if (p / "Dataset").exists() and (p / "Research_Paper").exists():
            return p
    # Fallback: parent of current file
    return start.parent


HERE = Path(__file__).resolve()
ROOT = find_project_root(HERE)

# Folders based on ROOT
DATASET_DIR = ROOT / "Dataset"
RESEARCH_PAPER_DIR = ROOT / "Research_Paper"
SCRIPTS_DIR = ROOT / "Script"
RAW_DIR = DATASET_DIR / "raw"
FINAL_DATASET_CSV = DATASET_DIR / "final_dataset.csv"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_PAPER_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Streamlit basic config
# --------------------------------------------------
st.set_page_config(
    page_title="Aerospace Alloy Dataset",
    layout="wide",
    page_icon="🛰️",
)


# --------------------------------------------------
# Pipeline scripts (must match your filenames)
# --------------------------------------------------
PIPELINE_SCRIPTS = [
    "00_scan_papers.py",
    "01_extract_raw_content.py",
    "02_clean_text.py",
    "03_extract_tables.py",
    "03b_extract_tables_camelot.py",
    "04_extract_Table_features.py",
    "05_extract_text_features.py",
    "06_build_final_dataset.py",
]


# --------------------------------------------------
# Helpers: clean data folders & run pipeline
# --------------------------------------------------
def reset_intermediate_data():
    """Remove intermediate folders and index/final files."""
    for sub in ["raw", "tables", "features"]:
        p = DATASET_DIR / sub
        if p.exists():
            shutil.rmtree(p)

    for name in [
        "papers_index.csv",
        "tables_index.csv",
        "tables_index.xlsx",
        "final_dataset.csv",
        "final_dataset.xlsx",
    ]:
        f = DATASET_DIR / name
        if f.exists():
            f.unlink()


def run_pipeline(clean: bool = False) -> str:
    """
    Run all processing scripts 00 -> 06.
    If clean=True, delete intermediate / old dataset first.
    Returns combined log text.
    """
    logs = []
    python_exe = sys.executable

    if clean:
        logs.append("=== CLEAN REBUILD: resetting intermediate data ===")
        try:
            reset_intermediate_data()
            logs.append("  -> raw/, tables/, features/, indexes, final_dataset removed.")
        except Exception as e:
            logs.append(f"[WARN] Could not fully reset intermediate data: {e}")
    else:
        logs.append("=== NORMAL RUN (no clean) ===")

    for script_name in PIPELINE_SCRIPTS:
        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            logs.append(f"[SKIP] {script_name} not found at {script_path}")
            continue

        logs.append(f"\n=== Running {script_name} ===")
        try:
            result = subprocess.run(
                [python_exe, str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(SCRIPTS_DIR),  # run inside Script/ folder
            )
            if result.stdout:
                logs.append(result.stdout)
            if result.stderr:
                logs.append("[stderr]")
                logs.append(result.stderr)
            if result.returncode != 0:
                logs.append(f"[ERROR] {script_name} exited with code {result.returncode}")
        except Exception as e:
            logs.append(f"[EXCEPTION] while running {script_name}: {e}")

    return "\n".join(logs)


# ===================== SIDEBAR =====================

with st.sidebar:
    st.header("🛰️ Appearance")
    st.caption("Use Streamlit's built-in light / dark modes up here.")

    st.markdown("---")
    st.header("📥 Data Ingestion")

    uploaded_files = st.file_uploader(
        "Upload research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        saved_files = []
        for up in uploaded_files:
            save_path = RESEARCH_PAPER_DIR / up.name
            with open(save_path, "wb") as f:
                f.write(up.getbuffer())
            saved_files.append(str(save_path))

        st.success(
            f"Saved {len(saved_files)} file(s) into:\n{RESEARCH_PAPER_DIR}"
        )

    st.markdown("---")
    st.header("⚙️ Build / Update Dataset")

    run_btn = st.button("Run pipeline (00 → 06)")
    clean_btn = st.button("Clean & run (delete old data, rebuild from PDFs)")

    st.markdown("---")
    st.caption(f"Project root: `{ROOT}`")
    st.caption(f"Papers folder: `{RESEARCH_PAPER_DIR}`")


# ===================== MAIN CONTENT =====================

st.title("Aerospace Alloy Dataset")
st.write(
    "Explore alloy compositions and thermo-mechanical properties extracted from research papers."
)

# Load existing dataset once at start (if present)
if "dataset" not in st.session_state and FINAL_DATASET_CSV.exists():
    st.session_state["dataset"] = pd.read_csv(FINAL_DATASET_CSV, encoding="utf-8")

# Decide which pipeline action to run
pipeline_action = None
if clean_btn:
    pipeline_action = "clean"
elif run_btn:
    pipeline_action = "normal"

if pipeline_action is not None:
    prev_rows = None
    if "dataset" in st.session_state:
        try:
            prev_rows = len(st.session_state["dataset"])
        except Exception:
            prev_rows = None

    with st.spinner(
        "Running extraction pipeline (clean rebuild)..." if pipeline_action == "clean"
        else "Running extraction pipeline..."
    ):
        log_text = run_pipeline(clean=(pipeline_action == "clean"))

    # Store log for optional viewing
    st.session_state["last_log"] = log_text

    # Reload dataset if exists
    if FINAL_DATASET_CSV.exists():
        df_new = pd.read_csv(FINAL_DATASET_CSV, encoding="utf-8")
        st.session_state["dataset"] = df_new
        new_rows = len(df_new)

        if prev_rows is None:
            st.success(f"Pipeline finished. Loaded dataset with {new_rows} rows.")
        elif prev_rows == new_rows:
            st.warning(
                f"Pipeline finished. Dataset still has {new_rows} rows "
                f"(no net change in feature rows)."
            )
        else:
            st.success(
                f"Pipeline finished. Rows changed: {prev_rows} → {new_rows}."
            )
    else:
        st.error(
            f"Pipeline finished, but {FINAL_DATASET_CSV} was not found.\n\n"
            "Most common causes:\n"
            "• No PDFs in Research_Paper folder\n"
            "• Feature extraction scripts (04 / 05) found no data\n"
            "• An earlier script failed (check the log below)"
        )

    # Optional: show log if user wants
    show_log = st.checkbox("Show pipeline log (advanced)")
    if show_log and "last_log" in st.session_state:
        st.expander("Pipeline log (00 → 06)", expanded=True).code(
            st.session_state["last_log"], language="text"
        )

# If still no dataset, stop
if "dataset" not in st.session_state:
    if FINAL_DATASET_CSV.exists():
        df = pd.read_csv(FINAL_DATASET_CSV, encoding="utf-8")
        st.session_state["dataset"] = df
        st.info(f"Loaded existing dataset with {len(df)} rows.")
    else:
        st.info("Upload PDFs and click a pipeline button in the sidebar.")
        st.stop()

df = st.session_state["dataset"].copy()

# ===================== FILTERS / SEARCH =====================

st.subheader("Dataset Filters and Search")
st.caption("Filter by paper ID, alloy name, composition, processing route, or properties.")

with st.expander("Column filters", expanded=True):
    all_columns = list(df.columns)
    default_show_cols = [
        "paper_id",
        "alloy_name",
        "composition_of_alloy",
        "density",
        "tensile_strength",
        "elongation",
        "thermal_conductivity",
        "thermal_expansion",
        "manufacturing_process",
    ]
    show_columns = st.multiselect(
        "Columns to display in the table",
        options=all_columns,
        default=[c for c in default_show_cols if c in all_columns],
    )

    st.markdown("#### Per-column search")
    filter_cols = [
        "paper_id",
        "alloy_name",
        "composition_of_alloy",
        "density",
        "tensile_strength",
        "elongation",
        "thermal_conductivity",
        "thermal_expansion",
        "manufacturing_process",
    ]
    filter_cols = [c for c in filter_cols if c in df.columns]

    col_left, col_right = st.columns(2)
    for i, col in enumerate(filter_cols):
        if i % 2 == 0:
            with col_left:
                q = st.text_input(f"Filter `{col}` contains:", key=f"filter_{col}")
        else:
            with col_right:
                q = st.text_input(f"Filter `{col}` contains:", key=f"filter_{col}")
        if q:
            df = df[df[col].astype(str).str.contains(q, case=False, na=False)]

    st.markdown("#### Global search")
    global_query = st.text_input(
        "Search anywhere (composition, properties, process, snippet):",
        key="global_search",
    )
    if global_query:
        key_cols = [
            "alloy_name",
            "composition_of_alloy",
            "density",
            "burn_factor",
            "extinction_pressure",
            "flammability_index",
            "tensile_strength",
            "elongation",
            "thermal_conductivity",
            "thermal_expansion",
            "manufacturing_process",
            "source_snippet",
        ]
        key_cols = [c for c in key_cols if c in df.columns]
        mask = pd.Series(False, index=df.index)
        for col in key_cols:
            mask |= df[col].astype(str).str.contains(global_query, case=False, na=False)
        df = df[mask]

st.write(f"Filtered rows: **{len(df)}**")

# ===================== TABLE VIEW =====================

st.subheader("Final Dataset")
if df.empty:
    st.warning("No rows match the current filters.")
    st.stop()

table_to_show = df[show_columns] if show_columns else df
st.dataframe(table_to_show, use_container_width=True)

# ===================== ROW DETAILS =====================

st.markdown("---")
st.subheader("Row details")

indices = list(table_to_show.index)
index_labels = [
    f"{idx} | "
    f"{table_to_show.loc[idx, 'paper_id'] if 'paper_id' in table_to_show.columns else ''} | "
    f"{table_to_show.loc[idx, 'composition_of_alloy'] if 'composition_of_alloy' in table_to_show.columns else ''}"
    for idx in indices
]

selected_idx = st.selectbox(
    "Choose a row to inspect",
    options=indices,
    index=0,
    format_func=lambda idx: index_labels[indices.index(idx)],
)

row = df.loc[selected_idx]

# ---------- Metadata ----------
st.markdown("### Metadata")
for col in [
    "paper_id",
    "pdf_path",
    "pdf_abs_path",
    "page_num",
    "table_idx",
    "row_idx",
    "source_type",
    "source_location",
]:
    if col in df.columns:
        st.write(f"**{col}**: {row.get(col, None)}")

if "pdf_abs_path" in row and isinstance(row["pdf_abs_path"], str):
    st.markdown(f"[Open PDF file (local path)](file://{row['pdf_abs_path']})")

# ---------- Composition and properties ----------
st.markdown("### Composition and Properties")
for col in [
    "alloy_name",
    "composition_of_alloy",
    "density",
    "burn_factor",
    "extinction_pressure",
    "flammability_index",
    "tensile_strength",
    "elongation",
    "thermal_conductivity",
    "thermal_expansion",
    "manufacturing_process",
]:
    if col in df.columns:
        st.write(f"**{col}**: {row.get(col, None)}")

# ---------- Source details ----------
st.markdown("### Source details")

if "source_type" in row:
    st.write(f"**Source type:** {row['source_type']}")

if "source_csv" in row and isinstance(row["source_csv"], str):
    st.write(f"**Source table CSV:** `{row['source_csv']}`")

if "source_snippet" in row and isinstance(row["source_snippet"], str):
    st.write("**Text snippet:**")
    st.caption(row["source_snippet"])

#source_type = str(row.get("source_type", "")).lower()

# ---- show original content (table or text) ----
source_type = str(row.get("source_type", "")).lower()

if source_type == "table" and isinstance(row.get("source_csv"), str):
    table_rel = row["source_csv"]
    table_path = ROOT / table_rel
    st.write("**Full table from which this row was extracted:**")
    if table_path.exists():
        try:
            df_table = pd.read_csv(table_path)
            st.dataframe(df_table, use_container_width=True)

            # Show the specific row that generated this feature row
            if pd.notna(row.get("row_idx")):
                try:
                    table_row_idx = int(row["row_idx"])
                    # row_idx is already the real index in df_table (0 = header,
                    # 1..N = data rows), so we do NOT subtract 1 here.
                    if table_row_idx in df_table.index:
                        st.write("**Selected row in table:**")
                        st.dataframe(
                            df_table.loc[[table_row_idx]],
                            use_container_width=True,
                        )
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"Could not read table CSV: {e}")
    else:
        st.warning(f"Table file not found at: {table_path}")


# If source is text, show the corresponding page text
elif source_type == "text":
    paper_id = row.get("paper_id")
    page_num = row.get("page_num")
    if isinstance(paper_id, str) and pd.notna(page_num):
        raw_path = RAW_DIR / f"{paper_id}_raw.json"
        st.write("**Source page text:**")
        if raw_path.exists():
            try:
                with raw_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                pages = data.get("pages", [])
                page_text = ""
                for p in pages:
                    if p.get("page_num") == int(page_num):
                        page_text = p.get("text", "") or ""
                        break
                if page_text:
                    st.text_area(
                        f"Page {int(page_num)} text",
                        page_text,
                        height=300,
                    )
                else:
                    st.info("Page text not found in raw JSON.")
            except Exception as e:
                st.warning(f"Could not read raw JSON: {e}")
        else:
            st.warning(f"Raw JSON file not found: {raw_path}")
