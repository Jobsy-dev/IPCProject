import re
from pathlib import Path
import pandas as pd

# ----------------- paths -----------------
ROOT = Path(__file__).resolve().parents[1]  # project root
DATASET_DIR = ROOT / "Dataset"
TABLES_INDEX_CSV = DATASET_DIR / "tables_index.csv"

FEATURES_DIR = DATASET_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = FEATURES_DIR / "Table_features.csv"
OUTPUT_XLSX = FEATURES_DIR / "Table_features.xlsx"


# ----------------- helper functions -----------------

def load_table(csv_rel_path: str) -> pd.DataFrame | None:
    csv_path = ROOT / csv_rel_path
    if not csv_path.exists():
        print(f"  [WARN] Table CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path, header=0, encoding="utf-8")
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(how="all")
    df = df.dropna(how="all", axis=1)

    if df.empty:
        return None
    return df


def find_column(header_series: pd.Series, keywords: list[str]) -> int | None:
    for idx, val in header_series.items():
        text = str(val).lower()
        for kw in keywords:
            if kw in text:
                return idx
    return None


def infer_tensile_type(header_text: str) -> str | None:
    h = str(header_text).lower()
    if "yield" in h or "0.2" in h or "ys" in h:
        return "YS"
    if "ultimate" in h or "uts" in h or "tensile" in h:
        return "UTS"
    return None


def infer_temperature_label(text: str) -> str | None:
    t = str(text).lower()
    if "room" in t or "rt" in t or "ambient" in t:
        return "RT"
    if "°c" in t or "deg c" in t or "elevated" in t or "high temp" in t:
        return "High"
    return None


def extract_temperature_numeric(text: str) -> float | None:
    """
    Extract a temperature from text and always return it in °C.

    Handles:
      - 25°C, 25 °C, 25 deg C
      - 77°F, 77 °F
      - 298K
      - plain numbers near 'temp' / 'temperature' / C / K / F context
    """
    if not isinstance(text, str):
        text = str(text)

    # ---- 1) explicit C / F / K with unit tokens ----
    unit = None
    m = re.search(r"([-+]?\d*\.?\d+)\s*(?:°\s*)?C\b", text, flags=re.IGNORECASE)
    if m:
        unit = "C"
    else:
        m = re.search(r"([-+]?\d*\.?\d+)\s*(?:deg\.?\s*C)", text, flags=re.IGNORECASE)
        if m:
            unit = "C"
    if not m:
        m = re.search(r"([-+]?\d*\.?\d+)\s*(?:°\s*)?F\b", text, flags=re.IGNORECASE)
        if m:
            unit = "F"
    if not m:
        m = re.search(r"([-+]?\d*\.?\d+)\s*K\b", text, flags=re.IGNORECASE)
        if m:
            unit = "K"

    if m and unit is not None:
        try:
            value = float(m.group(1))
        except ValueError:
            return None

        if unit == "C":
            return value
        elif unit == "F":
            # °F → °C
            return (value - 32.0) * 5.0 / 9.0
        elif unit == "K":
            # K → °C
            return value - 273.15

    # ---- 2) fallback: number near 'temp' / 'temperature' / C / K / F ----
    for mm in re.finditer(r"([-+]?\d*\.?\d+)", text):
        start, end = mm.span()
        context = text[max(0, start - 10): min(len(text), end + 10)].lower()

        if ("temp" in context or "temperature" in context or
                " °" in context or " c" in context or " k" in context or " f" in context):
            try:
                value = float(mm.group(1))
            except ValueError:
                continue

            if "fahrenheit" in context or "°f" in context or " f" in context:
                return (value - 32.0) * 5.0 / 9.0
            elif "kelvin" in context or " k" in context:
                return value - 273.15
            else:
                # assume °C
                return value

    return None


def infer_manufacturing_process(row_text: str) -> str | None:
    t = str(row_text).lower()
    if "cast" in t or "casting" in t:
        return "cast"
    if "wrought" in t:
        return "wrought"
    if "forged" in t or "forging" in t or "forge" in t:
        return "forged"
    if "rolled" in t or "rolling" in t:
        return "rolled"
    if "coating" in t or "coated" in t:
        return "coating"
    if "additive" in t or "3d printing" in t or "3d-printed" in t:
        return "additive_manufacturing"
    if "lpbf" in t or "laser powder bed fusion" in t or "slm" in t:
        return "lpbf"
    if "ded" in t or "directed energy deposition" in t:
        return "ded"
    return None


def extract_numeric(text: str) -> float | None:
    if not isinstance(text, str):
        text = str(text)

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not m:
        return None

    try:
        value = float(m.group(0))
    except ValueError:
        return None

    if value < 0:
        return None

    return value


def infer_tensile_unit(text: str) -> str | None:
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    if "mpa" in t:
        return "MPa"
    if "gpa" in t:
        return "GPa"
    if "ksi" in t:
        return "ksi"
    return None


def infer_elongation_unit(text: str) -> str | None:
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    if "%" in t or "percent" in t:
        return "%"
    return None


def infer_k_unit(text: str) -> str | None:
    """
    Infer unit for thermal conductivity.
    Handles:
      - 'W/mK', 'W/m*K'
      - 'W m-1 K-1'
      - 'W·m⁻¹·K⁻¹'
      - 'W m-1 °C-1'
    """
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()

    if "w/mk" in t or "w/m*k" in t:
        return "W/mK"

    if "w" in t and ("m-1" in t or "m^-1" in t or " m " in t or "w m" in t):
        if "k-1" in t or "k^-1" in t or "/k" in t:
            return "W/mK"
        if "c-1" in t or "°c-1" in t or "/°c" in t or "/c" in t:
            return "W/mK"

    if "conduct" in t and "w" in t:
        return "W/mK"

    return None


def infer_alpha_unit(text: str) -> str | None:
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    if ("10^-6" in t or "10-6" in t) and ("/k" in t or "k-1" in t or "k^-1" in t or "/°c" in t or "c-1" in t):
        return "10^-6/K"
    return None


def infer_density_unit(text: str) -> str | None:
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    if "kg/m3" in t or "kg m-3" in t:
        return "kg/m3"
    if "g/cm3" in t or "g cm-3" in t:
        return "g/cm3"
    return None


# ---------- alloy / composition helpers ----------

def looks_like_composition(text: str) -> bool:
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    if any(x in t for x in ["at.%", "wt.%", "at%", "wt%", "vol.%"]):
        return True
    if len(re.findall(r"\d+\.?\d*\s*%", t)) >= 2:
        return True
    return False


def clean_composition(raw: str) -> str | None:
    """
    Keep only 'Element + number + %' pieces, drop long sentences.
    """
    if not isinstance(raw, str):
        raw = str(raw)
    text = raw.strip()
    if not text:
        return None

    pattern = re.compile(
        r"[A-Z][a-z]?\s*[-:]?\s*\d+\.?\d*\s*(?:at\.?%|wt\.?%|vol\.?%|%)",
        flags=re.IGNORECASE,
    )
    parts = [m.group(0).strip() for m in pattern.finditer(text)]

    if parts:
        return ", ".join(parts)

    if len(text) <= 80 and looks_like_composition(text):
        return text

    return None


def parse_alloy_name_and_composition(cell_text: str) -> tuple[str | None, str | None]:
    """
    Used for the alloy-name column (e.g. 'GRCop-84 (Cu, 8 at% Cr, 4 at% Nb)').
    """
    if not isinstance(cell_text, str):
        cell_text = str(cell_text)

    txt = cell_text.strip()
    if not txt:
        return None, None

    low = txt.lower()
    if low.startswith("key:") or low.startswith("key ") or low.startswith("serve as"):
        return None, None

    if "(" in txt and ")" in txt and txt.index("(") < txt.index(")"):
        name_part = txt[: txt.index("(")].strip()
        comp_part_raw = txt[txt.index("(") + 1: txt.rindex(")")].strip()
        comp_clean = clean_composition(comp_part_raw)
        return (name_part or None), comp_clean

    return txt, None


def format_value_with_temp(value, unit, tempC, temp_label):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    try:
        v = f"{float(value):g}"
    except Exception:
        v = str(value)

    unit_part = f" {unit}" if unit else ""

    if tempC is not None and not (isinstance(tempC, float) and pd.isna(tempC)):
        try:
            t = f"{float(tempC):g}"
        except Exception:
            t = str(tempC)
        return f"{v}{unit_part} @ {t} °C"

    if isinstance(temp_label, str) and temp_label.strip():
        return f"{v}{unit_part} @ {temp_label.strip()}"

    return f"{v}{unit_part}"


# ---------- vertical property tables ----------

def is_vertical_property_table(df: pd.DataFrame) -> bool:
    if df.shape[1] != 2:
        return False

    col0_text = " ".join(df.iloc[:, 0].astype(str).head(8).tolist()).lower()
    keywords = [
        "density",
        "thermal conductivity",
        "coefficient of thermal expansion",
        "thermal expansion",
        "specific heat",
        "young",
        "poisson",
        "thermal diffusivity",
    ]
    if not any(kw in col0_text for kw in keywords):
        return False

    numeric_count = 0
    total = 0
    for v in df.iloc[:, 1]:
        total += 1
        if extract_numeric(v) is not None:
            numeric_count += 1
    if total == 0:
        return False

    return (numeric_count / total) >= 0.5


def process_vertical_property_table(
    paper_id: str,
    page_num: int,
    table_idx: int,
    csv_rel: str,
    df: pd.DataFrame,
    features_rows: list[dict],
):
    row_dict = {
        "paper_id": paper_id,
        "page_num": page_num,
        "table_idx": table_idx,
        "row_idx": 1,
        "source_csv": csv_rel,
        "alloy_name": None,
        "composition_of_alloy": None,
        "density_raw": "",
        "density_value": None,
        "density_unit": None,
        "tensile_strength_raw": "",
        "tensile_strength_value": None,
        "tensile_strength_type": None,
        "tensile_temperature_label": None,
        "tensile_temperature_C": None,
        "tensile_unit": None,
        "tensile_strength_with_temp": "",
        "elongation_raw": "",
        "elongation_value": None,
        "elongation_temperature_label": None,
        "elongation_temperature_C": None,
        "elongation_unit": None,
        "elongation_with_temp": "",
        "thermal_conductivity_raw": "",
        "thermal_conductivity_value": None,
        "thermal_conductivity_temperature_label": None,
        "thermal_conductivity_temperature_C": None,
        "thermal_conductivity_unit": None,
        "thermal_conductivity_with_temp": "",
        "thermal_expansion_raw": "",
        "thermal_expansion_value": None,
        "thermal_expansion_temperature_label": None,
        "thermal_expansion_temperature_C": None,
        "thermal_expansion_unit": None,
        "thermal_expansion_with_temp": "",
        "burn_factor_raw": "",
        "burn_factor_value": None,
        "extinction_pressure_raw": "",
        "extinction_pressure_value": None,
        "flammability_index_raw": "",
        "flammability_index_value": None,
        "manufacturing_process": None,
    }

    for _, r in df.iterrows():
        name = str(r.iloc[0])
        val_raw = str(r.iloc[1])
        val = extract_numeric(val_raw)
        name_lower = name.lower()

        if "density" in name_lower:
            row_dict["density_raw"] = f"{name}: {val_raw}"
            row_dict["density_value"] = val
            row_dict["density_unit"] = infer_density_unit(name)

        elif "thermal conductivity" in name_lower:
            row_dict["thermal_conductivity_raw"] = f"{name}: {val_raw}"
            row_dict["thermal_conductivity_value"] = val
            row_dict["thermal_conductivity_unit"] = infer_k_unit(name)

        elif "coefficient of thermal expansion" in name_lower or "thermal expansion" in name_lower or "cte" in name_lower:
            row_dict["thermal_expansion_raw"] = f"{name}: {val_raw}"
            row_dict["thermal_expansion_value"] = val
            row_dict["thermal_expansion_unit"] = infer_alpha_unit(name)

        elif "burn factor" in name_lower or "burning rate" in name_lower or "burn rate" in name_lower:
            row_dict["burn_factor_raw"] = f"{name}: {val_raw}"
            row_dict["burn_factor_value"] = val

        elif "extinguishing pressure" in name_lower or "extinction pressure" in name_lower \
                or "threshold pressure" in name_lower or "combustion pressure" in name_lower:
            row_dict["extinction_pressure_raw"] = f"{name}: {val_raw}"
            row_dict["extinction_pressure_value"] = val

        elif "flammability index" in name_lower or "flammability" in name_lower:
            row_dict["flammability_index_raw"] = f"{name}: {val_raw}"
            row_dict["flammability_index_value"] = val

    row_dict["thermal_conductivity_with_temp"] = format_value_with_temp(
        row_dict["thermal_conductivity_value"],
        row_dict["thermal_conductivity_unit"],
        row_dict["thermal_conductivity_temperature_C"],
        row_dict["thermal_conductivity_temperature_label"],
    )
    row_dict["thermal_expansion_with_temp"] = format_value_with_temp(
        row_dict["thermal_expansion_value"],
        row_dict["thermal_expansion_unit"],
        row_dict["thermal_expansion_temperature_C"],
        row_dict["thermal_expansion_temperature_label"],
    )

    features_rows.append(row_dict)


# ----------------- main extraction logic -----------------

def process_one_table(row_meta: dict, features_rows: list[dict]):
    csv_rel = row_meta["csv_path"]
    paper_id = row_meta["paper_id"]
    page_num = row_meta["page_num"]
    table_idx = row_meta["table_idx"]

    df = load_table(csv_rel)
    if df is None:
        return

    # vertical property list tables
    if is_vertical_property_table(df):
        print(f"  -> Detected vertical property table for {csv_rel}")
        process_vertical_property_table(
            paper_id, page_num, table_idx, csv_rel, df, features_rows
        )
        return

    header = df.iloc[0].fillna("").astype(str)
    data = df.iloc[1:].reset_index(drop=True)

    # separate alloy-name vs composition columns
    col_alloy = find_column(header, ["alloy name", "alloy id", "alloy", "material"])
    col_comp = find_column(header, ["chemical composition", "composition", "chemistry"])

    col_density = find_column(header, ["density", "kg/m3", "g/cm3"])
    col_burn = find_column(header, ["burn factor", "burning rate", "burn rate"])
    col_pressure = find_column(
        header,
        ["extinguishing", "threshold pressure", "extinction pressure",
         "combustion pressure", "p_c", "chamber pressure"]
    )
    col_flammability = find_column(header, ["flammability", "flammability index", "fi"])
    col_tensile = find_column(header, ["tensile", "ultimate tensile", "uts", "yield strength", "0.2%"])
    col_elong = find_column(header, ["elongation", "elong-ation", "elong", "el%", "strain to failure", "% elong"])
    col_k = find_column(header, ["thermal conductivity", "k (w/mk)", "conductivity"])
    col_alpha = find_column(header, ["thermal expansion", "cte", "coefficient of thermal expansion"])
    col_temp = find_column(header, ["temp", "temperature"])

    # ----- detect an explicit Celsius temperature column next to the temp column -----
    col_temp_C = None
    if col_temp is not None:
        try:
            temp_pos = df.columns.get_loc(col_temp)
        except KeyError:
            temp_pos = None
        if isinstance(temp_pos, int) and (temp_pos + 1) < df.shape[1]:
            next_col_label = df.columns[temp_pos + 1]
            next_col_vals = df[next_col_label].astype(str).str.lower()
            if next_col_vals.str.contains("°c").any() or next_col_vals.str.contains("deg c").any():
                col_temp_C = next_col_label

    # ---- temperature labels from headers ----
    tensile_temp_label_hint = infer_temperature_label(header[col_tensile]) if col_tensile is not None else None
    elong_temp_label_hint = infer_temperature_label(header[col_elong]) if col_elong is not None else None
    k_temp_label_hint = infer_temperature_label(header[col_k]) if col_k is not None else None

    # IMPORTANT: do NOT infer temperature label from CTE header (to avoid -6 °C from 10^-6)
    alpha_temp_label_hint = None

    # ---- numeric temperature hints from headers ----
    tensile_temp_C_hint = extract_temperature_numeric(header[col_tensile]) if col_tensile is not None else None
    elong_temp_C_hint = extract_temperature_numeric(header[col_elong]) if col_elong is not None else None
    k_temp_C_hint = extract_temperature_numeric(header[col_k]) if col_k is not None else None
    alpha_temp_C_hint = None  # disabled for CTE header

    tensile_type_hint = infer_tensile_type(header[col_tensile]) if col_tensile is not None else None

    tensile_unit_hint = infer_tensile_unit(header[col_tensile]) if col_tensile is not None else None
    elong_unit_hint = infer_elongation_unit(header[col_elong]) if col_elong is not None else None
    k_unit_hint = infer_k_unit(header[col_k]) if col_k is not None else None
    alpha_unit_hint = infer_alpha_unit(header[col_alpha]) if col_alpha is not None else None

    # -------- iterate over rows in the table --------
    for ridx, row in data.iterrows():
        row_text_join = " | ".join(str(x) for x in row.tolist())

        row_dict = {
            "paper_id": paper_id,
            "page_num": page_num,
            "table_idx": table_idx,
            "row_idx": int(ridx + 1),
            "source_csv": csv_rel,
        }

        # ----- temperature from columns (prefer explicit °C column if present) -----
        temp_val_C_from_col = None
        temp_label_from_col = None

        # 1) dedicated Celsius column (e.g. labeled °C) next to temperature column
        if col_temp_C is not None and col_temp_C in row.index:
            temp_raw_c = str(row[col_temp_C])
            if temp_raw_c and temp_raw_c.strip() and temp_raw_c.strip() != "-":
                temp_val_C_from_col = extract_temperature_numeric(temp_raw_c)
                if temp_val_C_from_col is None:
                    temp_val_C_from_col = extract_numeric(temp_raw_c)

        # 2) fall back to generic "temperature" column (could be °C, °F, or K)
        if temp_val_C_from_col is None and col_temp is not None and col_temp in row.index:
            temp_raw = str(row[col_temp])
            if temp_raw and temp_raw.strip() and temp_raw.strip() != "-":
                temp_val_C_from_col = extract_temperature_numeric(temp_raw)
                if temp_val_C_from_col is None:
                    temp_val_C_from_col = extract_numeric(temp_raw)
                temp_label_from_col = infer_temperature_label(temp_raw)

        # ----- alloy name & composition -----
        alloy_name = None
        composition = None

        if col_alloy is not None and col_alloy in row.index:
            name_raw = str(row[col_alloy])
            n, c_from_name = parse_alloy_name_and_composition(name_raw)
            alloy_name = n
            if c_from_name:
                composition = c_from_name

        if col_comp is not None and col_comp in row.index:
            comp_raw = str(row[col_comp])
            comp_clean = clean_composition(comp_raw)
            if comp_clean:
                composition = comp_clean

        row_dict["alloy_name"] = alloy_name
        row_dict["composition_of_alloy"] = composition

        # ---------- density ----------
        if col_density is not None and col_density in row.index:
            dens_raw = str(row[col_density])
            row_dict["density_raw"] = dens_raw
            row_dict["density_value"] = extract_numeric(dens_raw)
            row_dict["density_unit"] = infer_density_unit(header[col_density])
        else:
            row_dict["density_raw"] = ""
            row_dict["density_value"] = None
            row_dict["density_unit"] = None

        # ---------- burn factor ----------
        if col_burn is not None and col_burn in row.index:
            burn_raw = str(row[col_burn])
            row_dict["burn_factor_raw"] = burn_raw
            row_dict["burn_factor_value"] = extract_numeric(burn_raw)
        else:
            row_dict["burn_factor_raw"] = ""
            row_dict["burn_factor_value"] = None

        # ---------- extinction / combustion pressure ----------
        if col_pressure is not None and col_pressure in row.index:
            p_raw = str(row[col_pressure])
            row_dict["extinction_pressure_raw"] = p_raw
            row_dict["extinction_pressure_value"] = extract_numeric(p_raw)
        else:
            row_dict["extinction_pressure_raw"] = ""
            row_dict["extinction_pressure_value"] = None

        # ---------- flammability index ----------
        if col_flammability is not None and col_flammability in row.index:
            fl_raw = str(row[col_flammability])
            row_dict["flammability_index_raw"] = fl_raw
            row_dict["flammability_index_value"] = extract_numeric(fl_raw)
        else:
            row_dict["flammability_index_raw"] = ""
            row_dict["flammability_index_value"] = None

        # ---------- tensile strength ----------
        if col_tensile is not None and col_tensile in row.index:
            t_raw = str(row[col_tensile])
            row_dict["tensile_strength_raw"] = t_raw
            row_dict["tensile_strength_value"] = extract_numeric(t_raw)
            row_dict["tensile_strength_type"] = tensile_type_hint or infer_tensile_type(t_raw)

            t_label = tensile_temp_label_hint or infer_temperature_label(t_raw) or temp_label_from_col
            t_tempC = tensile_temp_C_hint if tensile_temp_C_hint is not None else extract_temperature_numeric(t_raw)
            if t_tempC is None:
                t_tempC = temp_val_C_from_col
            t_unit = tensile_unit_hint or infer_tensile_unit(t_raw)

            row_dict["tensile_temperature_label"] = t_label
            row_dict["tensile_temperature_C"] = t_tempC
            row_dict["tensile_unit"] = t_unit

            row_dict["tensile_strength_with_temp"] = format_value_with_temp(
                row_dict["tensile_strength_value"],
                row_dict["tensile_unit"],
                row_dict["tensile_temperature_C"],
                row_dict["tensile_temperature_label"],
            )
        else:
            row_dict["tensile_strength_raw"] = ""
            row_dict["tensile_strength_value"] = None
            row_dict["tensile_strength_type"] = None
            row_dict["tensile_temperature_label"] = None
            row_dict["tensile_temperature_C"] = None
            row_dict["tensile_unit"] = None
            row_dict["tensile_strength_with_temp"] = ""

        # ---------- elongation ----------
        if col_elong is not None and col_elong in row.index:
            e_raw = str(row[col_elong])
            row_dict["elongation_raw"] = e_raw
            row_dict["elongation_value"] = extract_numeric(e_raw)

            e_label = elong_temp_label_hint or infer_temperature_label(e_raw) or temp_label_from_col
            e_tempC = elong_temp_C_hint if elong_temp_C_hint is not None else extract_temperature_numeric(e_raw)
            if e_tempC is None:
                e_tempC = temp_val_C_from_col
            e_unit = elong_unit_hint or infer_elongation_unit(e_raw)

            row_dict["elongation_temperature_label"] = e_label
            row_dict["elongation_temperature_C"] = e_tempC
            row_dict["elongation_unit"] = e_unit

            row_dict["elongation_with_temp"] = format_value_with_temp(
                row_dict["elongation_value"],
                row_dict["elongation_unit"],
                row_dict["elongation_temperature_C"],
                row_dict["elongation_temperature_label"],
            )
        else:
            row_dict["elongation_raw"] = ""
            row_dict["elongation_value"] = None
            row_dict["elongation_temperature_label"] = None
            row_dict["elongation_temperature_C"] = None
            row_dict["elongation_unit"] = None
            row_dict["elongation_with_temp"] = ""

        # ---------- thermal conductivity ----------
        if col_k is not None and col_k in row.index:
            k_raw = str(row[col_k])
            row_dict["thermal_conductivity_raw"] = k_raw
            row_dict["thermal_conductivity_value"] = extract_numeric(k_raw)

            k_label = k_temp_label_hint or infer_temperature_label(k_raw) or temp_label_from_col
            k_tempC = k_temp_C_hint if k_temp_C_hint is not None else extract_temperature_numeric(k_raw)
            if k_tempC is None:
                k_tempC = temp_val_C_from_col

            k_unit = k_unit_hint or infer_k_unit(k_raw)
            if not k_unit:
                header_text = str(header[col_k]).lower()
                if "conduct" in header_text or "thermal conductivity" in header_text:
                    k_unit = "W/mK"
            if row_dict["thermal_conductivity_value"] is not None and k_unit is None:
                k_unit = "W/mK"

            row_dict["thermal_conductivity_temperature_label"] = k_label
            row_dict["thermal_conductivity_temperature_C"] = k_tempC
            row_dict["thermal_conductivity_unit"] = k_unit

            row_dict["thermal_conductivity_with_temp"] = format_value_with_temp(
                row_dict["thermal_conductivity_value"],
                row_dict["thermal_conductivity_unit"],
                row_dict["thermal_conductivity_temperature_C"],
                row_dict["thermal_conductivity_temperature_label"],
            )
        else:
            row_dict["thermal_conductivity_raw"] = ""
            row_dict["thermal_conductivity_value"] = None
            row_dict["thermal_conductivity_temperature_label"] = None
            row_dict["thermal_conductivity_temperature_C"] = None
            row_dict["thermal_conductivity_unit"] = None
            row_dict["thermal_conductivity_with_temp"] = ""

        # ---------- thermal expansion ----------
        if col_alpha is not None and col_alpha in row.index:
            a_raw = str(row[col_alpha])
            row_dict["thermal_expansion_raw"] = a_raw
            row_dict["thermal_expansion_value"] = extract_numeric(a_raw)

            # do NOT use alpha_temp_label_hint / alpha_temp_C_hint from header
            a_label = infer_temperature_label(a_raw) or temp_label_from_col
            a_tempC = extract_temperature_numeric(a_raw)
            if a_tempC is None:
                a_tempC = temp_val_C_from_col
            a_unit = alpha_unit_hint or infer_alpha_unit(a_raw)

            row_dict["thermal_expansion_temperature_label"] = a_label
            row_dict["thermal_expansion_temperature_C"] = a_tempC
            row_dict["thermal_expansion_unit"] = a_unit

            row_dict["thermal_expansion_with_temp"] = format_value_with_temp(
                row_dict["thermal_expansion_value"],
                row_dict["thermal_expansion_unit"],
                row_dict["thermal_expansion_temperature_C"],
                row_dict["thermal_expansion_temperature_label"],
            )
        else:
            row_dict["thermal_expansion_raw"] = ""
            row_dict["thermal_expansion_value"] = None
            row_dict["thermal_expansion_temperature_label"] = None
            row_dict["thermal_expansion_temperature_C"] = None
            row_dict["thermal_expansion_unit"] = None
            row_dict["thermal_expansion_with_temp"] = ""

        # ---------- manufacturing process ----------
        row_dict["manufacturing_process"] = infer_manufacturing_process(row_text_join)

        features_rows.append(row_dict)


def main():
    if not TABLES_INDEX_CSV.exists():
        print(f"tables_index.csv not found at: {TABLES_INDEX_CSV}")
        print("Run 03/03b_extract_tables first.")
        return

    df_index = pd.read_csv(TABLES_INDEX_CSV, encoding="utf-8")
    print(f"Loaded {len(df_index)} table entries from tables_index.csv")

    # ------------- INCREMENTAL: load existing features (if any) -------------
    features_rows: list[dict] = []
    processed_tables = set()

    if OUTPUT_CSV.exists():
        df_old = pd.read_csv(OUTPUT_CSV, encoding="utf-8")
        if not df_old.empty:
            features_rows.extend(df_old.to_dict(orient="records"))
            processed_tables = set(
                zip(
                    df_old["paper_id"],
                    df_old["page_num"],
                    df_old["table_idx"],
                )
            )
            print(
                f"Loaded {len(df_old)} existing feature rows "
                f"from {OUTPUT_CSV} "
                f"for {len(processed_tables)} table(s)."
            )
        else:
            print(f"{OUTPUT_CSV} exists but is empty – rebuilding features.")
    else:
        print("No existing Table_features.csv – extracting for all tables.")

    new_feature_rows = 0

    # ------------- process only new tables -------------
    for _, meta_row in df_index.iterrows():
        key = (meta_row["paper_id"], int(meta_row["page_num"]), int(meta_row["table_idx"]))
        if key in processed_tables:
            print(
                f"[SKIP] Features already exist for "
                f"paper={key[0]}, page={key[1]}, table={key[2]}"
            )
            continue

        row_meta = {
            "paper_id": meta_row["paper_id"],
            "page_num": int(meta_row["page_num"]),
            "table_idx": int(meta_row["table_idx"]),
            "csv_path": meta_row["csv_path"],
        }
        print(
            f"Processing tables for paper={row_meta['paper_id']}, "
            f"page={row_meta['page_num']}, table={row_meta['table_idx']} ..."
        )

        before = len(features_rows)
        process_one_table(row_meta, features_rows)
        after = len(features_rows)
        added = after - before
        new_feature_rows += added
        print(f"  -> Added {added} feature row(s) for this table.")

    if not features_rows:
        print("No feature rows extracted – check your keyword mappings.")
        return

    df_features = pd.DataFrame(features_rows)
    df_features.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    df_features.to_excel(OUTPUT_XLSX, index=False)

    print(f"\nNew feature rows extracted this run: {new_feature_rows}")
    print(f"Total material-feature rows now: {len(df_features)}")
    print(f"CSV written to:   {OUTPUT_CSV}")
    print(f"Excel written to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
