import json
import re
from pathlib import Path
import pandas as pd

# -------- paths --------
ROOT = Path(__file__).resolve().parents[1]  # project root
RAW_DIR = ROOT / "Dataset" / "raw"
FEATURES_DIR = ROOT / "Dataset" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = FEATURES_DIR / "text_features.csv"
OUTPUT_XLSX = FEATURES_DIR / "text_features.xlsx"


# ---------- numeric + temperature helpers ----------
def clean_composition(raw: str) -> str | None:
    """
    Keep only proper 'Element + number + %' pieces, drop long sentences.

    Examples:
      'Cu, 8 at% Cr, 4 at% Nb'  -> 'Cu 8 at%, Cr 4 at%, Nb ?' (but we only keep 'element+%'-like parts)
      'Cu-8 at.% Cr-4 at.% Nb' -> 'Cu-8 at.%, Cr-4 at.%'
      Long sentence with no clear composition -> None
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

    # if there are no clear element+% pieces but the string is short and
    # already looks like a compact composition, keep it; otherwise drop.
    if len(text) <= 80 and any(x in text.lower() for x in ["at.%", "wt.%", "at%", "wt%", "vol.%", "%"]):
        return text

    return None


def extract_numeric(text: str):
    """
    Extract the first float-like number from text.
    For this dataset (MPa, %, W/mK, etc.), we ignore clearly negative values
    to avoid things like '-42' from IDs.
    """
    if not isinstance(text, str):
        text = str(text)

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not m:
        return None

    try:
        value = float(m.group(0))
    except ValueError:
        return None

    # physical properties should not be negative here
    if value < 0:
        return None

    return value


def infer_temperature_from_text(text: str):
    """Rough temperature label (RT/High) from surrounding text."""
    t = text.lower()
    if "room temperature" in t or "rt" in t or "ambient" in t:
        return "RT"
    if "°c" in t or "deg c" in t or "° c" in t or "high temperature" in t or "elevated" in t:
        return "High"
    return None


def extract_temperature_numeric(text: str) -> float | None:
    """
    Extract a temperature from text and always return it in °C.

    Handles:
      - 25°C, 25 °C, 25 deg C
      - 77°F, 77 °F
      - 298K
      - plain numbers near 'temp' / 'temperature' / K / F context
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

            # infer unit from local context if possible
            if "fahrenheit" in context or "°f" in context or " f" in context:
                return (value - 32.0) * 5.0 / 9.0
            elif "kelvin" in context or " k" in context:
                return value - 273.15
            else:
                # default: assume °C
                return value

    # ---- 3) no reliable temperature ----
    return None


def infer_tensile_type_from_text(text: str):
    """Roughly decide if tensile is UTS or YS from surrounding words."""
    t = text.lower()
    if "yield" in t or "0.2" in t or "ys" in t:
        return "YS"
    if "ultimate" in t or "uts" in t or "tensile strength" in t:
        return "UTS"
    return None


# ---------- unit helpers ----------

def infer_density_unit(text: str, explicit_unit: str | None) -> str | None:
    if explicit_unit:
        u = explicit_unit.lower()
        if "kg/m3" in u or "kg m-3" in u:
            return "kg/m3"
        if "g/cm3" in u or "g cm-3" in u:
            return "g/cm3"
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    if "kg/m3" in t or "kg m-3" in t:
        return "kg/m3"
    if "g/cm3" in t or "g cm-3" in t:
        return "g/cm3"
    return None


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
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    if "w/mk" in t or "w m-1 k-1" in t or "w·m" in t:
        return "W/mK"
    return None


def infer_alpha_unit(text: str) -> str | None:
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    if "10^-6" in t or "10-6" in t:
        return "10^-6/K"
    if "/k" in t or "1/k" in t:
        return "1/K"
    return None


def make_snippet(text: str, start: int, end: int, window: int = 80) -> str:
    """Take a small snippet around the match indices."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right]
    return " ".join(snippet.split())  # collapse whitespace


def format_value_with_temp(
    value: float | None,
    unit: str | None,
    tempC: float | None,
    temp_label: str | None,
) -> str:
    """
    Build a nice string like '650 MPa @ 25 °C' or '10 % @ RT'.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    # value
    try:
        val_str = f"{float(value):g}"
    except Exception:
        val_str = str(value)

    unit_part = f" {unit}" if unit else ""

    # Prefer numeric temperature if available
    if tempC is not None and not (isinstance(tempC, float) and pd.isna(tempC)):
        try:
            t_str = f"{float(tempC):g}"
        except Exception:
            t_str = str(tempC)
        return f"{val_str}{unit_part} @ {t_str} °C"

    # Otherwise use label (RT, High, etc.)
    if temp_label:
        return f"{val_str}{unit_part} @ {temp_label}"

    # No temperature information
    return f"{val_str}{unit_part}"


def base_empty_row(paper_id, page_num, snippet):
    """
    Base dict with all shared keys so every rows.append has same columns.
    alloy_name / composition_of_alloy are here (usually empty for text matches).
    """
    return {
        "paper_id": paper_id,
        "page_num": page_num,
        "source_type": "text",
        "source_location": "page_text",
        "source_snippet": snippet,

        "alloy_name": None,
        "composition_of_alloy": None,

        "density_raw": "",
        "density_value": None,
        "density_unit": None,

        "burn_factor_raw": "",
        "burn_factor_value": None,

        "extinction_pressure_raw": "",
        "extinction_pressure_value": None,

        "flammability_index_raw": "",
        "flammability_index_value": None,

        "tensile_strength_raw": "",
        "tensile_strength_value": None,
        "tensile_strength_type": None,
        "tensile_temperature_condition": None,
        "tensile_temperature_C": None,
        "tensile_unit": None,
        "tensile_strength_with_temp": "",

        "elongation_raw": "",
        "elongation_percent": None,
        "elongation_temperature_condition": None,
        "elongation_temperature_C": None,
        "elongation_unit": None,
        "elongation_with_temp": "",

        "thermal_conductivity_raw": "",
        "thermal_conductivity_value": None,
        "thermal_conductivity_temperature_condition": None,
        "thermal_conductivity_temperature_C": None,
        "thermal_conductivity_unit": None,
        "thermal_conductivity_with_temp": "",

        "thermal_expansion_raw": "",
        "thermal_expansion_value": None,
        "thermal_expansion_temperature_condition": None,
        "thermal_expansion_temperature_C": None,
        "thermal_expansion_unit": None,
        "thermal_expansion_with_temp": "",

        "manufacturing_process": None,

        "table_idx": None,
        "row_idx": None,
        "source_csv": None,
    }


def process_page_text(paper_id: str, page_num: int, page_text: str, rows: list[dict]):
    if not page_text:
        return

    text = page_text

    # ---- Alloy name + composition in parentheses ----
    comp_paren_pattern = re.compile(
        r"([A-Za-z0-9\-]+)\s*\(([^)]*(?:at%|wt\.?%|at\.%|wt%|%) [^)]*)\)",
        flags=re.IGNORECASE,
    )
    for m in comp_paren_pattern.finditer(text):
        alloy_name = m.group(1).strip()
        raw_comp = m.group(2).strip()

        # clean the composition (remove long sentences, keep only element+% parts)
        composition = clean_composition(raw_comp)
        if not composition:
            continue  # skip if it's not a real composition

        snippet = make_snippet(text, m.start(), m.end())

        row = base_empty_row(paper_id, page_num, snippet)
        row["alloy_name"] = alloy_name
        row["composition_of_alloy"] = composition
        rows.append(row)

    # ---- Density ----
    density_pattern = re.compile(
        r"(density[^.\n]{0,80}?)"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"[^A-Za-z0-9]{0,10}(kg/m3|kg m-3|g/cm3|g cm-3)?",
        flags=re.IGNORECASE,
    )
    for m in density_pattern.finditer(text):
        context = m.group(0)
        value_str = m.group(2)
        unit_str = m.group(3)
        val = extract_numeric(value_str)
        unit = infer_density_unit(context, unit_str)
        snippet = make_snippet(text, m.start(), m.end())

        row = base_empty_row(paper_id, page_num, snippet)
        row["density_raw"] = context.strip()
        row["density_value"] = val
        row["density_unit"] = unit

        rows.append(row)

    # ---- Tensile strength (UTS/YS) ----
    tensile_pattern = re.compile(
        r"(tensile strength[^.\n]{0,80}?|ultimate tensile[^.\n]{0,80}?|yield strength[^.\n]{0,80}?)"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        flags=re.IGNORECASE,
    )
    for m in tensile_pattern.finditer(text):
        ctx = m.group(0)
        val = extract_numeric(m.group(2))
        snippet = make_snippet(text, m.start(), m.end())
        t_type = infer_tensile_type_from_text(ctx)
        t_temp_label = infer_temperature_from_text(ctx)
        t_temp_C = extract_temperature_numeric(ctx)
        t_unit = infer_tensile_unit(ctx)

        row = base_empty_row(paper_id, page_num, snippet)
        row["tensile_strength_raw"] = ctx.strip()
        row["tensile_strength_value"] = val
        row["tensile_strength_type"] = t_type
        row["tensile_temperature_condition"] = t_temp_label
        row["tensile_temperature_C"] = t_temp_C
        row["tensile_unit"] = t_unit
        row["tensile_strength_with_temp"] = format_value_with_temp(
            val, t_unit, t_temp_C, t_temp_label
        )

        rows.append(row)

    # ---- Elongation ----
    elong_pattern = re.compile(
        r"(elongation[^.\n]{0,80}?|strain to failure[^.\n]{0,80}?|% elong[^.\n]{0,80}?)"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*%",
        flags=re.IGNORECASE,
    )
    for m in elong_pattern.finditer(text):
        ctx = m.group(0)
        val = extract_numeric(m.group(2))
        snippet = make_snippet(text, m.start(), m.end())
        e_temp_label = infer_temperature_from_text(ctx)
        e_temp_C = extract_temperature_numeric(ctx)
        e_unit = infer_elongation_unit(ctx)

        row = base_empty_row(paper_id, page_num, snippet)
        row["elongation_raw"] = ctx.strip()
        row["elongation_percent"] = val
        row["elongation_temperature_condition"] = e_temp_label
        row["elongation_temperature_C"] = e_temp_C
        row["elongation_unit"] = e_unit
        row["elongation_with_temp"] = format_value_with_temp(
            val, e_unit, e_temp_C, e_temp_label
        )

        rows.append(row)

    # ---- Thermal conductivity ----
    k_pattern = re.compile(
        r"(thermal conductivity[^.\n]{0,80}?|conductivity[^.\n]{0,80}?)"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        flags=re.IGNORECASE,
    )
    for m in k_pattern.finditer(text):
        ctx = m.group(0)
        val = extract_numeric(m.group(2))
        snippet = make_snippet(text, m.start(), m.end())
        k_temp_label = infer_temperature_from_text(ctx)
        k_temp_C = extract_temperature_numeric(ctx)
        k_unit = infer_k_unit(ctx)

        row = base_empty_row(paper_id, page_num, snippet)
        row["thermal_conductivity_raw"] = ctx.strip()
        row["thermal_conductivity_value"] = val
        row["thermal_conductivity_temperature_condition"] = k_temp_label
        row["thermal_conductivity_temperature_C"] = k_temp_C
        row["thermal_conductivity_unit"] = k_unit
        row["thermal_conductivity_with_temp"] = format_value_with_temp(
            val, k_unit, k_temp_C, k_temp_label
        )

        rows.append(row)

    # ---- Thermal expansion ----
    alpha_pattern = re.compile(
        r"(thermal expansion[^.\n]{0,80}?|coefficient of thermal expansion[^.\n]{0,80}?|cte[^.\n]{0,80}?)"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        flags=re.IGNORECASE,
    )
    for m in alpha_pattern.finditer(text):
        ctx = m.group(0)
        val = extract_numeric(m.group(2))
        snippet = make_snippet(text, m.start(), m.end())
        a_temp_label = infer_temperature_from_text(ctx)
        a_temp_C = extract_temperature_numeric(ctx)
        a_unit = infer_alpha_unit(ctx)

        row = base_empty_row(paper_id, page_num, snippet)
        row["thermal_expansion_raw"] = ctx.strip()
        row["thermal_expansion_value"] = val
        row["thermal_expansion_temperature_condition"] = a_temp_label
        row["thermal_expansion_temperature_C"] = a_temp_C
        row["thermal_expansion_unit"] = a_unit
        row["thermal_expansion_with_temp"] = format_value_with_temp(
            val, a_unit, a_temp_C, a_temp_label
        )

        rows.append(row)

    # ---- Burn factor ----
    burn_pattern = re.compile(
        r"(burn factor[^.\n]{0,80}?|burning rate[^.\n]{0,80}?|burn rate[^.\n]{0,80}?)"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        flags=re.IGNORECASE,
    )
    for m in burn_pattern.finditer(text):
        ctx = m.group(0)
        val = extract_numeric(m.group(2))
        snippet = make_snippet(text, m.start(), m.end())

        row = base_empty_row(paper_id, page_num, snippet)
        row["burn_factor_raw"] = ctx.strip()
        row["burn_factor_value"] = val

        rows.append(row)

    # ---- Extinguishing / combustion pressure ----
    p_pattern = re.compile(
        r"(extinguishing pressure[^.\n]{0,80}?|threshold pressure[^.\n]{0,80}?|extinction pressure[^.\n]{0,80}?|combustion pressure[^.\n]{0,80}?)"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        flags=re.IGNORECASE,
    )
    for m in p_pattern.finditer(text):
        ctx = m.group(0)
        val = extract_numeric(m.group(2))
        snippet = make_snippet(text, m.start(), m.end())

        row = base_empty_row(paper_id, page_num, snippet)
        row["extinction_pressure_raw"] = ctx.strip()
        row["extinction_pressure_value"] = val

        rows.append(row)

    # ---- Flammability index ----
    flamm_pattern = re.compile(
        r"(flammability index[^.\n]{0,80}?|flammability[^.\n]{0,80}?)"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        flags=re.IGNORECASE,
    )
    for m in flamm_pattern.finditer(text):
        ctx = m.group(0)
        val = extract_numeric(m.group(2))
        snippet = make_snippet(text, m.start(), m.end())

        row = base_empty_row(paper_id, page_num, snippet)
        row["flammability_index_raw"] = ctx.strip()
        row["flammability_index_value"] = val

        rows.append(row)


def process_raw_file(raw_path: Path, rows: list[dict]):
    with raw_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    paper_id = data.get("paper_id", raw_path.stem.replace("_raw", ""))
    pages = data.get("pages", [])

    print(f"Processing text for paper: {paper_id} (pages: {len(pages)})")

    for page in pages:
        page_num = page.get("page_num")
        text = page.get("text", "") or ""
        process_page_text(paper_id, page_num, text, rows)


def main():
    raw_files = sorted(RAW_DIR.glob("*_raw.json"))
    if not raw_files:
        print(f"No *_raw.json files found in {RAW_DIR}")
        return

    all_rows: list[dict] = []
    processed_papers = set()
    new_rows_count = 0

    # -------- incremental: load existing text_features (if any) --------
    if OUTPUT_CSV.exists():
        try:
            df_old = pd.read_csv(OUTPUT_CSV, encoding="utf-8")
            if not df_old.empty:
                all_rows.extend(df_old.to_dict(orient="records"))
                if "paper_id" in df_old.columns:
                    processed_papers = set(df_old["paper_id"].astype(str).unique())
                print(
                    f"Loaded {len(df_old)} existing text-feature rows "
                    f"from {OUTPUT_CSV} for {len(processed_papers)} paper(s)."
                )
        except Exception as e:
            print(f"[WARN] Could not read existing {OUTPUT_CSV}: {e}")

    # -------- process only new papers --------
    for raw_path in raw_files:
        paper_id = raw_path.stem.replace("_raw", "")
        if paper_id in processed_papers:
            print(f"[SKIP] Text-features already exist for paper_id={paper_id}")
            continue

        before = len(all_rows)
        process_raw_file(raw_path, all_rows)
        after = len(all_rows)
        added = after - before
        new_rows_count += added
        print(f"  -> Added {added} new text-feature row(s) for {paper_id}")

    if not all_rows:
        print("No text-based features extracted. Check patterns.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    df.to_excel(OUTPUT_XLSX, index=False)

    print(f"\nNew text-feature rows this run: {new_rows_count}")
    print(f"Total text-feature rows now: {len(df)}")
    print(f"CSV written to:   {OUTPUT_CSV}")
    print(f"Excel written to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
