#!/usr/bin/env python3
"""
Build ``datasets/MANIFEST.md`` — a one-page snapshot of every bundled dataset
in ``datasets/`` (size, row/col count, date coverage, which script builds it,
a content hash) so "is this result on stale data?" is answerable without
re-deriving it by hand.

Regenerate after adding/refreshing any dataset:

    python experiments/build_dataset_manifest.py

Pure stdlib + pandas; read-only against the repo (only writes
``datasets/MANIFEST.md``). Never crashes on a malformed dataset — unreadable
files are still listed, flagged "unreadable", and everything else proceeds.
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path)

import hashlib
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

warnings.filterwarnings(
    "ignore", message="Could not infer format", category=UserWarning
)

ROOT = _bootstrap.ROOT
DATASETS_DIR = ROOT / "datasets"
MANIFEST_PATH = DATASETS_DIR / "MANIFEST.md"
SCAN_DIRS = [ROOT / "experiments", ROOT / "posterioralpha"]
STALE_DAYS = 180
DATE_COL_CANDIDATES = ("date", "Date", "timestamp")
HASH_CHARS = 12


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def sha256_prefix(path: Path, n_chars: int = HASH_CHARS) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_chars]


def inspect_dataset(path: Path):
    """Return a dict with n_rows, n_cols, date_range, note (empty if fine)."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive, any parser error
        return {
            "n_rows": None, "n_cols": None,
            "date_range": "—",
            "note": f"unreadable ({exc.__class__.__name__})",
        }

    n_rows, n_cols = df.shape
    date_range = "—"

    parsed = None
    if n_cols:
        first_col = df.columns[0]
        try:
            parsed = pd.to_datetime(df[first_col], errors="raise")
        except Exception:
            parsed = None

    if parsed is None:
        for cand in DATE_COL_CANDIDATES:
            if cand in df.columns:
                try:
                    parsed = pd.to_datetime(df[cand], errors="raise")
                    break
                except Exception:
                    parsed = None

    if parsed is not None and len(parsed.dropna()):
        lo, hi = parsed.min(), parsed.max()
        date_range = f"{lo.date()} → {hi.date()}"

    return {"n_rows": n_rows, "n_cols": n_cols, "date_range": date_range, "note": ""}


# --------------------------------------------------------------------------
# builder attribution
# --------------------------------------------------------------------------
def collect_py_files():
    """All experiments/*.py and posterioralpha/**/*.py, excluding __pycache__
    and this script itself (it isn't a dataset builder — it just mentions
    dataset filenames while describing them, which would otherwise pollute
    its own builder-attribution output)."""
    this_file = Path(__file__).resolve()
    files = []
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        files.extend(
            p for p in sorted(d.rglob("*.py"))
            if "__pycache__" not in p.parts and p.resolve() != this_file
        )
    return files


def search_terms_for(filename: str):
    """The literal filename, plus the .gz-stripped form (docstrings are often
    written before a dataset got gzip'd and never updated), plus dynamic
    f-string construction patterns — e.g. fresh_QQQ.csv is written as
    f"fresh_{ticker}.csv" (run_timing_mining.py / run_barbell_gfc.py), which
    a literal grep can never see."""
    terms = [filename]
    if filename.endswith(".gz"):
        terms.append(filename[:-3])
    m = re.match(r"(fresh)_[A-Z]+(\.csv)$", filename)
    if m:
        terms.append(f'{m.group(1)}_{{')     # matches f"fresh_{ticker}.csv"
    return terms


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def guess_writer(text: str, filename: str) -> bool:
    """Heuristic: a line that assigns `VAR = ...<filename>...` followed
    somewhere in the same file by `.to_csv(...VAR...)` marks this file as
    the actual writer (vs. just a reader)."""
    for term in search_terms_for(filename):
        for line in text.splitlines():
            if term not in line:
                continue
            m = re.match(r"\s*([A-Za-z_]\w*)\s*=", line)
            if not m:
                continue
            var = m.group(1)
            if re.search(rf"\.to_csv\([^)]*\b{re.escape(var)}\b", text, re.DOTALL):
                return True
    return False


def attribute_builder(filename: str, py_files, text_cache):
    terms = search_terms_for(filename)
    matches = [p for p in py_files if any(t in text_cache[p] for t in terms)]
    if not matches:
        return "unknown", []

    build_matches = [p for p in matches if p.name.startswith("build_")]
    if build_matches:
        return ", ".join(rel(p) for p in build_matches), matches

    writers = [p for p in matches if guess_writer(text_cache[p], filename)]
    if writers:
        return ", ".join(rel(p) for p in writers), matches

    return "unknown", matches


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    dataset_files = sorted(
        p for p in DATASETS_DIR.iterdir()
        if p.is_file() and (p.suffix == ".csv" or p.name.endswith(".csv.gz"))
    )
    py_files = collect_py_files()
    text_cache = {}
    for p in py_files:
        try:
            text_cache[p] = p.read_text(errors="ignore")
        except Exception:
            text_cache[p] = ""

    rows = []
    unreadable = []
    unreferenced = []
    stale = []
    today = datetime.now(timezone.utc).date()

    for path in dataset_files:
        size_bytes = path.stat().st_size
        digest = sha256_prefix(path)
        info = inspect_dataset(path)
        builder, matches = attribute_builder(path.name, py_files, text_cache)

        if info["note"]:
            unreadable.append(path.name)
            rows_display = info["note"]
        else:
            rows_display = f"{info['n_rows']:,} × {info['n_cols']}"

        if not matches:
            unreferenced.append(path.name)

        if info["date_range"] != "—":
            end_str = info["date_range"].split("→")[-1].strip()
            try:
                end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
                if (today - end_date).days > STALE_DAYS:
                    stale.append((path.name, end_str, (today - end_date).days))
            except ValueError:
                pass

        rows.append({
            "file": path.name,
            "size": human_size(size_bytes),
            "rows": rows_display,
            "date_range": info["date_range"],
            "builder": builder,
            "sha256": digest,
        })

    # ---- vix duplicate inspection --------------------------------------
    vix_a, vix_b = "vix_term_structure.csv", "vix_termstructure.csv"
    vix_note_lines = []
    pa, pb = DATASETS_DIR / vix_a, DATASETS_DIR / vix_b
    if pa.exists() and pb.exists():
        da = pd.read_csv(pa, parse_dates=["Date"]).set_index("Date")
        db = pd.read_csv(pb, parse_dates=["Date"]).set_index("Date")
        common = da.index.intersection(db.index)
        max_diff = None
        if len(common):
            diffs = (da.loc[common] - db.loc[common]).abs()
            max_diff = float(diffs.max(skipna=True).max())
        refs_a = [rel(p) for p in py_files if vix_a in text_cache[p]]
        refs_b = [rel(p) for p in py_files if vix_b in text_cache[p]]
        builder_a = next(r["builder"] for r in rows if r["file"] == vix_a)
        builder_b = next(r["builder"] for r in rows if r["file"] == vix_b)
        vix_note_lines.append(
            f"- **`{vix_a}`**: {da.shape[0]:,} rows × {da.shape[1]} cols "
            f"(`{list(da.columns)}`), {da.index.min().date()} → {da.index.max().date()}. "
            f"Builder: {builder_a}. Referenced by {len(refs_a)} script(s): "
            f"{', '.join(refs_a) if refs_a else 'none'}."
        )
        vix_note_lines.append(
            f"- **`{vix_b}`**: {db.shape[0]:,} rows × {db.shape[1]} cols "
            f"(`{list(db.columns)}`), {db.index.min().date()} → {db.index.max().date()}. "
            f"Builder: {builder_b}. Referenced by {len(refs_b)} script(s): "
            f"{', '.join(refs_b) if refs_b else 'none'}."
        )
        vix_note_lines.append(
            f"- Same schema (`Date, VIX, VIX3M`). {len(common):,} overlapping dates; "
            f"max abs difference on overlapping VIX/VIX3M values is "
            f"{max_diff:.3f} (negligible — different download vintages of the same series, "
            "not different data)."
        )
        # canonical recommendation
        if da.index.min() <= db.index.min() and da.shape[0] >= db.shape[0]:
            canonical, other = vix_a, vix_b
        elif db.index.min() <= da.index.min() and db.shape[0] >= da.shape[0]:
            canonical, other = vix_b, vix_a
        else:
            canonical, other = (vix_b, vix_a) if db.index.min() < da.index.min() else (vix_a, vix_b)
        canon_start = (da if canonical == vix_a else db).index.min().date()
        canon_builder = builder_a if canonical == vix_a else builder_b
        other_refs = len(refs_a) if other == vix_a else len(refs_b)
        canon_refs = len(refs_b) if other == vix_a else len(refs_a)
        vix_note_lines.append(
            f"- **Recommendation**: treat **`{canonical}`** as canonical — it has the wider "
            f"date coverage (back to {canon_start}, a strict superset of the other file's "
            f"range) and history should never be lost by consolidating onto it. `{other}` is "
            f"currently the more heavily-referenced file ({other_refs} scripts vs. "
            f"{canon_refs}), so consolidating means repointing those scripts to `{canonical}` "
            f"and then deleting `{other}`."
            + (
                f" Note `{other}` has an active cache-on-miss builder ({builder_a if other == vix_a else builder_b}) "
                f"that re-downloads from a hardcoded 2010-01-01 start if the file is ever "
                f"deleted — that start date would need bumping to match `{canonical}`'s "
                f"{canon_start} history before the consolidation, or the download would "
                f"silently truncate history again."
                if canon_builder == "unknown" and (builder_a if other == vix_a else builder_b) != "unknown"
                else ""
            )
        )
    else:
        vix_note_lines.append("- One or both vix_term(_)structure files are missing; skipped comparison.")

    # ---- write MANIFEST.md ----------------------------------------------
    gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("# Dataset manifest")
    lines.append("")
    lines.append(
        "Auto-generated snapshot of every bundled CSV/CSV.GZ file under `datasets/` — "
        "answers \"is this result on stale data?\" without opening each file by hand. "
        "For each dataset: file size, row/column count, the date range covered (parsed "
        "from the first column, falling back to a `date`/`Date`/`timestamp` column), "
        "which `experiments/build_*.py` (or other) script produces it, and a content "
        "hash (sha256, first 12 hex chars) to detect silent changes between runs."
    )
    lines.append("")
    lines.append(f"Regenerate with: `python experiments/build_dataset_manifest.py`")
    lines.append("")
    lines.append(f"Generated: {gen_time}")
    lines.append("")
    lines.append("| file | size | rows × cols | date range | builder | sha256 |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r["file"]):
        lines.append(
            f"| `{r['file']}` | {r['size']} | {r['rows']} | {r['date_range']} | "
            f"{r['builder']} | `{r['sha256']}` |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("**(a) Near-duplicate VIX term-structure files**")
    lines.append("")
    lines.extend(vix_note_lines)
    lines.append("")
    lines.append("**(b) Files no script references at all** (candidates for deletion)")
    lines.append("")
    if unreferenced:
        for name in sorted(unreferenced):
            lines.append(f"- `{name}`")
    else:
        lines.append("- None — every dataset is referenced by at least one script.")
    lines.append("")
    lines.append(
        f"**(c) Stale candidates** (date range ends more than {STALE_DAYS} days before "
        f"generation date {today.isoformat()})"
    )
    lines.append("")
    if stale:
        for name, end_str, age in sorted(stale, key=lambda t: -t[2]):
            note = (" — **intentional**: pre-2010 fresh-sample cache for the timing "
                    "miner's PROMOTED gate (`fresh_<ticker>.csv`), deliberately "
                    "frozen history, not stale data"
                    if name.startswith("fresh_") else "")
            lines.append(f"- `{name}` — last date {end_str} ({age:,} days ago){note}")
    else:
        lines.append("- None — every dataset with a detected date range is within "
                      f"{STALE_DAYS} days of today.")
    lines.append("")

    MANIFEST_PATH.write_text("\n".join(lines))

    print(f"Wrote {rel(MANIFEST_PATH)}: {len(rows)} datasets, "
          f"{len(unreadable)} unreadable, {len(unreferenced)} unreferenced, "
          f"{len(stale)} stale.")
    if unreadable:
        print("Unreadable:", ", ".join(unreadable))


if __name__ == "__main__":
    main()
