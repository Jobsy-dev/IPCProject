import pandas as pd
from pathlib import Path
import re

# -------- paths --------
ROOT = Path(__file__).resolve().parents[1]  # project root

DATASET_DIR = ROOT / "Dataset"

FEATURES_TABLE_CSV = DATASET_DIR / "features" / "Table_features.csv"
FEATURES_TEXT_CSV = DATASET_DIR / "features" / "text_features.csv"
PAPERS_INDEX_CSV = DATASET_DIR / "papers_index.csv"

OUTPUT_CSV = DATASET_DIR / "final_dataset.csv"
OUTPUT_XLSX = DATASET_DIR / "final_dataset.xlsx"


# ---------- feature string builder ----------
def combine_feature(
    row,
    value_col: str | None = None,
    unit_col: str | None = None,
    temp_C_col: str | None = None,
    temp_label_col: str | None = None,
    type_col: str | None = None,
    raw_temp_col: str | None = None,
):
    """
    Build a short feature string like:
      - '650 MPa @ 25 °C (UTS)'
      - '10 % @ RT'
    (We only show °C in the final string; F/K are converted earlier.)
    """
    pieces: list[str] = []

    # ---- numeric value ----
    if value_col and value_col in row and pd.notna(row[value_col]):
        pieces.append(str(row[value_col]))

    # ---- unit ----
    if unit_col and unit_col in row:
        u = row[unit_col]
        if isinstance(u, str) and u.strip():
            if pieces:
                pieces[-1] = f"{pieces[-1]} {u.strip()}"
            else:
                pieces.append(u.strip())

    # ---- temperature: numeric °C from column ----
    tempC_val = None
    tempC_str = None

    if temp_C_col and temp_C_col in row and pd.notna(row[temp_C_col]):
        try:
            tempC_val = float(row[temp_C_col])
            tempC_str = f"{tempC_val:g} °C"
        except Exception:
            tempC_val = None
            tempC_str = None

    # ---- textual temperature label (RT, High, etc.) ----
    temp_label = None
    if temp_label_col and temp_label_col in row:
        tlabel = row[temp_label_col]
        if isinstance(tlabel, str) and tlabel.strip():
            temp_label = tlabel.strip()

    # ---- decide final temperature string (only °C or label) ----
    temp_str = None
    if tempC_str:
        temp_str = tempC_str
    elif temp_label:
        temp_str = temp_label

    if temp_str:
        pieces.append(f"@ {temp_str}")

    # ---- type (UTS / YS) ----
    if type_col and type_col in row:
        t = row[type_col]
        if isinstance(t, str) and t.strip():
            pieces.append(f"({t.strip()})")

    main = " ".join(pieces).strip()
    return main if main else None


# ---------- helpers ----------
def has_numeric_value(s: str | None) -> bool:
    """Return True if string contains at least one digit."""
    if s is None or not isinstance(s, str):
        return False
    return bool(re.search(r"\d+(\.\d+)?", s))


def is_empty_value(x) -> bool:
    """Treat NaN or empty/whitespace strings as empty."""
    if pd.isna(x):
        return True
    if isinstance(x, str) and not x.strip():
        return True
    return False


# ---------- MAIN ----------
def main():
    # ---- load feature sources ----
    df_list = []

    if FEATURES_TABLE_CSV.exists():
        print(f"Loading table features from: {FEATURES_TABLE_CSV}")
        df_tab = pd.read_csv(FEATURES_TABLE_CSV, encoding="utf-8")
        df_tab["source_type"] = "table"
        df_list.append(df_tab)
    else:
        print(f"[WARN] Table features not found: {FEATURES_TABLE_CSV}")

    if FEATURES_TEXT_CSV.exists():
        print(f"Loading text features from: {FEATURES_TEXT_CSV}")
        df_txt = pd.read_csv(FEATURES_TEXT_CSV, encoding="utf-8")
        df_txt["source_type"] = "text"
        df_list.append(df_txt)
    else:
        print(f"[WARN] Text features not found: {FEATURES_TEXT_CSV}")

    if not df_list:
        print("No feature CSVs found. Run extraction first.")
        return

    df_feat = pd.concat(df_list, ignore_index=True)

    # alias for elongation_value
    if "elongation_value" not in df_feat.columns and "elongation_percent" in df_feat.columns:
        df_feat["elongation_value"] = df_feat["elongation_percent"]

    print(f"Total feature rows: {len(df_feat)}")

    # ---- load paper index ----
    if not PAPERS_INDEX_CSV.exists():
        print(f"ERROR: papers_index.csv not found at: {PAPERS_INDEX_CSV}")
        return

    df_papers = pd.read_csv(PAPERS_INDEX_CSV, encoding="utf-8")

    # ---- merge to attach pdf_path ----
    df = df_feat.merge(df_papers, on="paper_id", how="left")
    df["pdf_abs_path"] = (ROOT / "").as_posix() + "/" + df["pdf_path"].astype(str)

    # ---------- NEW: drop rows where tensile strength unit is ksi ----------
    if "tensile_unit" in df.columns:
        mask_ksi = df["tensile_unit"].astype(str).str.lower().str.contains("ksi")
        before_ksi = len(df)
        df = df[~mask_ksi].copy()
        print(f"Removed rows with tensile unit 'ksi': {before_ksi - len(df)}")

    # ---------- build combined feature strings ----------
    df["density"] = df.apply(
        lambda r: combine_feature(r, "density_value", "density_unit"),
        axis=1,
    )

    df["burn_factor"] = df.apply(
        lambda r: combine_feature(r, "burn_factor_value"),
        axis=1,
    )

    df["extinction_pressure"] = df.apply(
        lambda r: combine_feature(r, "extinction_pressure_value"),
        axis=1,
    )

    df["flammability_index"] = df.apply(
        lambda r: combine_feature(r, "flammability_index_value"),
        axis=1,
    )

    df["tensile_strength"] = df.apply(
        lambda r: combine_feature(
            r,
            value_col="tensile_strength_value",
            unit_col="tensile_unit",
            temp_C_col="tensile_temperature_C",
            temp_label_col=(
                "tensile_temperature_label"
                if "tensile_temperature_label" in df.columns
                else "tensile_temperature_condition"
            ),
            type_col="tensile_strength_type",
            raw_temp_col="tensile_strength_raw",
        ),
        axis=1,
    )

    df["elongation"] = df.apply(
        lambda r: combine_feature(
            r,
            value_col="elongation_value",
            unit_col="elongation_unit",
            temp_C_col="elongation_temperature_C",
            temp_label_col=(
                "elongation_temperature_label"
                if "elongation_temperature_label" in df.columns
                else "elongation_temperature_condition"
            ),
            raw_temp_col="elongation_raw",
        ),
        axis=1,
    )

    df["thermal_conductivity"] = df.apply(
        lambda r: combine_feature(
            r,
            value_col="thermal_conductivity_value",
            unit_col="thermal_conductivity_unit",
            temp_C_col="thermal_conductivity_temperature_C",
            temp_label_col=(
                "thermal_conductivity_temperature_label"
                if "thermal_conductivity_temperature_label" in df.columns
                else "thermal_conductivity_temperature_condition"
            ),
            raw_temp_col="thermal_conductivity_raw",
        ),
        axis=1,
    )

    df["thermal_expansion"] = df.apply(
        lambda r: combine_feature(
            r,
            value_col="thermal_expansion_value",
            unit_col="thermal_expansion_unit",
            temp_C_col="thermal_expansion_temperature_C",
            temp_label_col=(
                "thermal_expansion_temperature_label"
                if "thermal_expansion_temperature_label" in df.columns
                else "thermal_expansion_temperature_condition"
            ),
            raw_temp_col="thermal_expansion_raw",
        ),
        axis=1,
    )

    # ----------------------------------------------------------
    # STRICT FILTER:
    # Must have at least ONE numeric value or composition.
    # ----------------------------------------------------------
    numeric_cols = [
        "density_value",
        "burn_factor_value",
        "extinction_pressure_value",
        "flammability_index_value",
        "tensile_strength_value",
        "elongation_value",
        "thermal_conductivity_value",
        "thermal_expansion_value",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    numeric_mask = (
        df[numeric_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
        if numeric_cols
        else pd.Series(False, index=df.index)
    )

    comp_mask = (
        df["composition_of_alloy"].apply(has_numeric_value)
        if "composition_of_alloy" in df.columns
        else pd.Series(False, index=df.index)
    )

    df = df[numeric_mask | comp_mask].copy()
    print(f"Rows after strict numeric filtering: {len(df)}")

    # ---- select only required columns ----
    columns_final = [
        "paper_id",
        "pdf_path",
        "pdf_abs_path",
        "page_num",
        "table_idx",
        "row_idx",
        "source_type",
        "source_location",
        "source_snippet",
        "source_csv",
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
    ]
    columns_final = [c for c in columns_final if c in df.columns]
    df_final = df[columns_final].copy()

    # ----------------------------------------------------------
    # Remove rows where ALL key feature columns are empty
    # ----------------------------------------------------------
    key_feature_cols = [
        "alloy_name",
        "composition_of_alloy",
        "density",
        "tensile_strength",
        "elongation",
        "thermal_conductivity",
        "thermal_expansion",
        "manufacturing_process",
    ]
    key_feature_cols = [c for c in key_feature_cols if c in df_final.columns]

    if key_feature_cols:
        empty_mask = df_final[key_feature_cols].applymap(is_empty_value).all(axis=1)
        before_rows = len(df_final)
        df_final = df_final[~empty_mask].copy()
        print(f"Removed empty-feature rows: {before_rows - len(df_final)}")

    # ---- save ----
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    df_final.to_excel(OUTPUT_XLSX, index=False)

    print(f"\nFinal dataset rows: {len(df_final)}")
    print(f"CSV written to:   {OUTPUT_CSV}")
    print(f"Excel written to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
