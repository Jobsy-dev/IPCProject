import streamlit as st
import pandas as pd
from pathlib import Path
import json

# ================================
# Paths
# ================================
HERE = Path(__file__).resolve()
ROOT = HERE.parent
DATASET_DIR = ROOT / "Dataset"
RAW_DIR = DATASET_DIR / "raw"
FINAL_DATASET_CSV = DATASET_DIR / "final_dataset.csv"

# ================================
# Streamlit config
# ================================
st.set_page_config(
    page_title="Aerospace Alloy Dataset",
    layout="wide",
    page_icon="🛰️",
)

st.title("🛰 Aerospace Alloy Dataset")
st.caption(
    "Viewer for `Dataset/final_dataset.csv` with source tables/text. "
    "The heavy extraction pipeline runs only on your local machine."
)

# ================================
# Load dataset
# ================================
@st.cache_data
def load_dataset(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8")

df = load_dataset(FINAL_DATASET_CSV)

if df is None:
    st.error(
        "I couldn't find `Dataset/final_dataset.csv`.\n\n"
        "Generate it locally (with your scripts or `app1.py`) and commit it "
        "to GitHub at exactly that path."
    )
    st.stop()

st.success(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

# ================================
# Filters & Explorer
# ================================
st.subheader("Filters and search")

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
    "Columns to display",
    options=all_columns,
    default=[c for c in default_show_cols if c in all_columns],
)

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

filtered = df.copy()

col_left, col_right = st.columns(2)
for i, col in enumerate(filter_cols):
    container = col_left if i % 2 == 0 else col_right
    with container:
        val = st.text_input(f"`{col}` contains:", key=f"filter_{col}")
    if val:
        filtered = filtered[
            filtered[col].astype(str).str.contains(val, case=False, na=False)
        ]

global_query = st.text_input(
    "Global search (composition, properties, process, snippet, etc.):",
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
    key_cols = [c for c in key_cols if c in filtered.columns]
    mask = pd.Series(False, index=filtered.index)
    for col in key_cols:
        mask |= filtered[col].astype(str).str.contains(
            global_query, case=False, na=False
        )
    filtered = filtered[mask]

st.write(f"Filtered rows: **{len(filtered)}**")

st.subheader("Dataset")
table_to_show = filtered[show_columns] if show_columns else filtered
st.dataframe(table_to_show, use_container_width=True, height=400)

# ================================
# Row details + source view
# ================================
st.subheader("Row details")

if table_to_show.empty:
    st.info("No rows match the current filters.")
    st.stop()

indices = list(table_to_show.index)
index_labels = [
    f"{idx} | "
    f"{table_to_show.loc[idx, 'paper_id'] if 'paper_id' in table_to_show.columns else ''} | "
    f"{table_to_show.loc[idx, 'composition_of_alloy'] if 'composition_of_alloy' in table_to_show.columns else ''}"
    for idx in indices
]

selected_idx = st.selectbox(
    "Choose a row",
    options=indices,
    index=0,
    format_func=lambda idx: index_labels[indices.index(idx)],
)

row = df.loc[selected_idx]

col_meta, col_props = st.columns(2)

with col_meta:
    st.markdown("**Metadata**")
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

with col_props:
    st.markdown("**Composition and properties**")
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

st.markdown("---")
st.markdown("### Source details")

if "source_snippet" in row and isinstance(row["source_snippet"], str):
    st.write("**Text snippet:**")
    st.caption(row["source_snippet"])

source_type = str(row.get("source_type", "")).lower()

# ---- Source = table ----
if source_type == "table" and isinstance(row.get("source_csv"), str):
    # Normalise Windows-style paths (backslashes) to POSIX-style for Linux
    raw_rel = str(row["source_csv"])
    table_rel = raw_rel.replace("\\", "/").lstrip("./")

    table_path = ROOT / table_rel
    st.write("**Full table from which this row was extracted:**")

    if table_path.exists():
        try:
            df_table = pd.read_csv(table_path)
            st.dataframe(df_table, use_container_width=True)

            if pd.notna(row.get("row_idx")):
                try:
                    table_row_idx = int(row["row_idx"])
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

# ---- Source = text (raw JSON) ----
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
