#!/usr/bin/python3
# =============================================================================
# run_pipeline.py  —  Orchestrator for the FFT-scorer interface analysis pipeline
# =============================================================================
#
# OVERVIEW
# --------
# Single entry point that drives multi_interface.run_interface() across all
# three run modes for every complex in PRDBv3.json:
#
#   MODE A — "complex"   (bound source-truth)
#       Input : ALL_PDBs/<C_PDB>/<C_PDB>.pdb
#       Chains: C_pro_chain  (protein)  /  C_RNA_chain  (RNA)
#       Output: ALL_PDBs/<C_PDB>/complex_results/
#
#   MODE B — "unbound"   (individual unbound structures)
#       Input : ALL_PDBs/<C_PDB>/<U_pro_PDB>.pdb  (protein, U_PRO_chain)
#               ALL_PDBs/<C_PDB>/<U_RNA_PDB>.pdb  (RNA,     U_RNA_chain)
#       Only runs when Docking_case == "UU" (both unbound structures available).
#       U_RNA_PDB values may carry a trailing '*' — stripped automatically.
#       Output: ALL_PDBs/<C_PDB>/unbound_results/
#
#   MODE C — "generated"  (FFT docked outputs, all ranks)
#       Input : generated_PDBS/<C_PDB>/rank<N>/protein.pdb
#               generated_PDBS/<C_PDB>/rank<N>/rna.pdb
#       Output: generated_PDBS/<C_PDB>/rank<N>/results/
#
# MULTI-CHAIN HANDLING
# --------------------
# C_pro_chain and C_RNA_chain may contain multiple characters (e.g. "ABKM", "MN").
# Each character is a separate chain ID.  run_interface() is called once for
# every (protein_chain, rna_chain) pair, matching the original List_1.txt
# behaviour.  Results are stored under the key "<pro_chain><rna_chain>",
# e.g. {"AE": {...}, "BE": {...}, "KE": {...}, "ME": {...}}.
#
# USAGE
# -----
#   python run_pipeline.py \
#       --json      D:\FFT-scorer\assets\PRDBv3.json \
#       --truth_dir D:\FFT-scorer\assets\ALL_PDBs \
#       --gen_dir   D:\FFT-scorer\src\generated_PDBS \
#       --out_dir   D:\FFT-scorer\src\results
#
#   Optional flags:
#       --naccess_path  <path>   Full path to naccess binary if not on PATH
#                                e.g. /usr/local/bin/naccess
#       --skip_complex           Skip bound complex analysis (Mode A)
#       --skip_unbound           Skip unbound structure analysis (Mode B)
#       --skip_generated         Skip generated/docked analysis (Mode C)
#
# OUTPUT FILES (in --out_dir)
# ---------------------------
#   pipeline_results.json    machine-readable dict of all run_interface() returns
#   pipeline.log             full run log (mirrors to stdout)
#   bsa_comparison.tsv  \
#   residue_comparison.tsv   produced by compare_results.py (auto-called)
#   summary.tsv         /
# =============================================================================

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# Locate multi_interface.py — both files live in the same directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import multi_interface as mi
import compare_results as cr


# ---------------------------------------------------------------------------
# Logging — stdout + pipeline.log in out_dir
# ---------------------------------------------------------------------------

def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "pipeline.log")
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout),
        ]
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_star(pdb_id: str) -> str:
    """
    Strip a trailing '*' from a PDB ID.
    e.g. "3KNH*" → "3KNH"
    The '*' in PRDBv3.json marks RNAs extracted from a larger assembly.
    The file on disk has no asterisk.
    """
    return pdb_id.rstrip('*') if isinstance(pdb_id, str) else pdb_id


def expand_chains(chain_str: str) -> list:
    """
    Convert a chain string to a list of individual single-letter chain IDs.
    e.g. "ABKM" → ["A", "B", "K", "M"]
         "A"    → ["A"]
         "MN"   → ["M", "N"]
    Each character in the string is a separate chain ID — this matches the
    convention used in PRDBv3.json and the original List_1.txt format.
    """
    if not chain_str:
        return []
    return list(str(chain_str))


def pdb_folder(truth_dir: str, complex_id: str) -> str:
    """Return ALL_PDBs/<complex_id>/ path."""
    return os.path.join(truth_dir, complex_id)


def rank_folders(gen_dir: str, complex_id: str) -> list:
    """
    Return a numerically sorted list of rank sub-folder paths for a complex.
    Expects folders named rank1, rank2, … inside generated_PDBS/<complex_id>/.
    Returns an empty list if the complex folder does not exist.
    """
    base = os.path.join(gen_dir, complex_id)
    if not os.path.isdir(base):
        return []
    folders = sorted(
        [
            os.path.join(base, d)
            for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
            and d.lower().startswith("rank")
        ],
        key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or '0')
    )
    return folders


def skipped_result(reason: str) -> dict:
    """Standard sentinel dict for a skipped run."""
    return {
        "skipped": True, "reason": reason,
        "bsa_complex": "NA", "bsa_pro": "NA", "bsa_rna": "NA",
        "pro_int": "NA", "rna_int": "NA", "combined_int": "NA",
        "has_interface": False, "error": None,
    }


def safe_run(label: str, fn, *args, **kwargs) -> dict:
    """
    Call fn(*args, **kwargs).  Catch any exception and return a standardised
    error result so one bad PDB never aborts the whole pipeline.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        msg = f"EXCEPTION in {label}: {exc}\n{traceback.format_exc()}"
        logging.error(msg)
        return {
            "bsa_complex": "NA", "bsa_pro": "NA", "bsa_rna": "NA",
            "pro_int": "NA", "rna_int": "NA", "combined_int": "NA",
            "has_interface": False, "error": msg,
        }


# ---------------------------------------------------------------------------
# naccess path injection
# ---------------------------------------------------------------------------

def patch_naccess_path(naccess_path: str) -> None:
    """
    If the user supplies --naccess_path, prepend its directory to PATH so
    that os.system("naccess ...") calls inside multi_interface.py find the
    binary without the user having to install it system-wide.
    """
    if not naccess_path:
        return
    naccess_path = os.path.abspath(naccess_path)
    if not os.path.isfile(naccess_path):
        logging.warning(
            f"--naccess_path '{naccess_path}' does not exist — "
            f"falling back to system PATH"
        )
        return
    naccess_dir = os.path.dirname(naccess_path)
    os.environ["PATH"] = naccess_dir + os.pathsep + os.environ.get("PATH", "")
    logging.info(f"naccess binary set to: {naccess_path}")


def check_naccess_available() -> bool:
    """
    Return True if naccess is reachable on the current PATH.
    Logs a clear warning if not found, so the user knows exactly what to fix.
    """
    import shutil
    if shutil.which("naccess") is not None:
        logging.info("naccess found on PATH — OK")
        return True
    logging.error(
        "naccess NOT found on PATH.\n"
        "  naccess is a Linux-only binary. Options:\n"
        "  1. Run this pipeline inside WSL (Windows Subsystem for Linux):\n"
        "       - Install WSL: `wsl --install` in PowerShell (as Administrator)\n"
        "       - Access your files at /mnt/d/FFT-scorer/...\n"
        "       - Install naccess inside WSL, then run the pipeline with python3\n"
        "  2. Provide the full path with --naccess_path /path/to/naccess\n"
        "  3. Run on a Linux server / HPC cluster\n"
        "  The pipeline will CONTINUE but all BSA values will be NA until\n"
        "  naccess is available."
    )
    return False


# ---------------------------------------------------------------------------
# Mode A — bound complex (source truth)
# ---------------------------------------------------------------------------

def run_complex_mode(entry: dict, truth_dir: str) -> dict:
    """
    Run run_interface() in "complex" mode for one PRDBv3 entry.

    Handles multi-character chain strings (e.g. C_pro_chain="ABKM") by
    iterating over every (protein_chain × rna_chain) pair and running
    run_interface() once per pair.

    Returns
    -------
    dict keyed by "<pro_chain><rna_chain>", e.g. {"AR": {...}, "BR": {...}}
    Single-chain entries return a dict with one key, e.g. {"AR": {...}}.
    """
    c_pdb       = entry["C_PDB"]
    input_dir   = pdb_folder(truth_dir, c_pdb)
    results_dir = os.path.join(input_dir, "complex_results")
    source_pdb  = os.path.join(input_dir, c_pdb + ".pdb")

    # Validate source file exists before any chain iteration
    if not os.path.isfile(source_pdb):
        msg = f"[complex] Source PDB not found: {source_pdb}"
        logging.warning(msg)
        return {"_error": skipped_result(msg)}

    # Expand potentially multi-character chain strings into individual IDs
    pro_chains = expand_chains(entry.get("C_pro_chain", ""))
    rna_chains = expand_chains(entry.get("C_RNA_chain", ""))

    if not pro_chains or not rna_chains:
        msg = f"[complex] {c_pdb} — missing chain IDs (pro={pro_chains}, rna={rna_chains})"
        logging.warning(msg)
        return {"_error": skipped_result(msg)}

    pair_results = {}

    for pro_ch in pro_chains:
        for rna_ch in rna_chains:
            pair_key = f"{pro_ch}{rna_ch}"
            logging.info(
                f"[complex] {c_pdb}  pair={pair_key}  "
                f"protein='{pro_ch}'  rna='{rna_ch}'  input={input_dir}"
            )
            result = safe_run(
                f"complex/{c_pdb}/{pair_key}",
                mi.run_interface,
                pdb_file    = c_pdb,
                first_chain = pro_ch,
                second_chain= rna_ch,
                run_mode    = "complex",
                input_dir   = input_dir,
                results_dir = results_dir,
            )
            logging.info(
                f"[complex] {c_pdb}/{pair_key}  BSA={result['bsa_complex']}  "
                f"has_interface={result['has_interface']}  error={result.get('error')}"
            )
            pair_results[pair_key] = result

    return pair_results


# ---------------------------------------------------------------------------
# Mode B — unbound structures
# ---------------------------------------------------------------------------

def run_unbound_mode(entry: dict, truth_dir: str) -> dict:
    """
    Run run_interface() in "unbound" mode for one PRDBv3 entry.

    Gating:
        Docking_case == "UU"  → both unbound available → run
        Docking_case == "UB"  → RNA has no unbound     → skip
        Docking_case == "BU"  → protein has no unbound → skip

    U_PRO_chain and U_RNA_chain are single-character fields in the JSON
    (unbound structures always reference one chain each), so no multi-chain
    expansion is needed here.

    Returns
    -------
    dict  — single-entry result (unbound always produces one chain pair), or
            a skipped-sentinel dict wrapped under "_skipped" key.
    """
    c_pdb        = entry["C_PDB"]
    docking_case = entry.get("Docking_case", "")

    # Gate: only UU has both unbound structures
    if docking_case != "UU":
        msg = (f"[unbound] {c_pdb}  skipped — Docking_case='{docking_case}' "
               f"(need 'UU')")
        logging.info(msg)
        return {"_skipped": skipped_result(msg)}

    u_pro_pdb = strip_star(entry.get("U_pro_PDB"))
    u_rna_pdb = strip_star(entry.get("U_RNA_PDB"))
    u_pro_chain = entry.get("U_PRO_chain")
    u_rna_chain = entry.get("U_RNA_chain")

    # Validate all required fields are present
    missing_fields = [f for f, v in [
        ("U_pro_PDB",  u_pro_pdb),
        ("U_RNA_PDB",  u_rna_pdb),
        ("U_PRO_chain", u_pro_chain),
        ("U_RNA_chain", u_rna_chain),
    ] if not v]

    if missing_fields:
        msg = f"[unbound] {c_pdb}  skipped — missing fields: {missing_fields}"
        logging.warning(msg)
        return {"_skipped": skipped_result(msg)}

    folder      = pdb_folder(truth_dir, c_pdb)
    pro_path    = os.path.join(folder, u_pro_pdb + ".pdb")
    rna_path    = os.path.join(folder, u_rna_pdb + ".pdb")
    results_dir = os.path.join(folder, "unbound_results")

    # Validate files exist on disk
    missing_files = [p for p in [pro_path, rna_path] if not os.path.isfile(p)]
    if missing_files:
        msg = (f"[unbound] {c_pdb}  skipped — "
               f"PDB file(s) not found: {missing_files}")
        logging.warning(msg)
        return {"_skipped": skipped_result(msg)}

    pair_key = f"{u_pro_chain}{u_rna_chain}"
    logging.info(
        f"[unbound] {c_pdb}  pair={pair_key}  "
        f"pro={u_pro_pdb}(chain '{u_pro_chain}')  "
        f"rna={u_rna_pdb}(chain '{u_rna_chain}')"
    )

    result = safe_run(
        f"unbound/{c_pdb}/{pair_key}",
        mi.run_interface,
        pdb_file    = c_pdb,
        first_chain = u_pro_chain,
        second_chain= u_rna_chain,
        run_mode    = "unbound",
        input_dir   = folder,
        results_dir = results_dir,
        pre_split   = {"protein": pro_path, "rna": rna_path},
    )
    logging.info(
        f"[unbound] {c_pdb}/{pair_key}  BSA={result['bsa_complex']}  "
        f"has_interface={result['has_interface']}  error={result.get('error')}"
    )
    return {pair_key: result}


# ---------------------------------------------------------------------------
# Mode C — generated / docked structures (all ranks)
# ---------------------------------------------------------------------------

def run_generated_mode(entry: dict, gen_dir: str) -> dict:
    """
    Run run_interface() in "generated" mode for every rank folder of one complex.

    For generated PDBs, protein.pdb and rna.pdb are already split files, so
    chain IDs from the JSON are used only for output file naming (not extraction).
    The first protein chain and first RNA chain from C_pro_chain / C_RNA_chain
    are used as the naming stem (e.g. "A" and "R" → files named "1ASY_A.int").

    Returns
    -------
    dict keyed by rank label: {"rank1": {...}, "rank2": {...}, ...}
    """
    c_pdb  = entry["C_PDB"]
    ranks  = rank_folders(gen_dir, c_pdb)

    if not ranks:
        logging.warning(
            f"[generated] {c_pdb}  no rank folders found under "
            f"{os.path.join(gen_dir, c_pdb)}"
        )
        return {}

    # For output file naming only — use first chain of each set
    pro_chains = expand_chains(entry.get("C_pro_chain", ""))
    rna_chains = expand_chains(entry.get("C_RNA_chain", ""))
    naming_pro = pro_chains[0] if pro_chains else "P"
    naming_rna = rna_chains[0] if rna_chains else "R"

    rank_results = {}

    for rank_path in ranks:
        rank_label = os.path.basename(rank_path)   # e.g. "rank1"
        pro_path   = os.path.join(rank_path, "protein.pdb")
        rna_path   = os.path.join(rank_path, "rna.pdb")

        # Validate both docked files exist
        missing_files = [p for p in [pro_path, rna_path]
                         if not os.path.isfile(p)]
        if missing_files:
            msg = (f"[generated] {c_pdb}/{rank_label}  skipped — "
                   f"file(s) not found: {missing_files}")
            logging.warning(msg)
            rank_results[rank_label] = skipped_result(msg)
            continue

        logging.info(
            f"[generated] {c_pdb}/{rank_label}  "
            f"input={rank_path}  output={rank_path}/results"
        )
        result = safe_run(
            f"generated/{c_pdb}/{rank_label}",
            mi.run_interface,
            pdb_file    = c_pdb,
            first_chain = naming_pro,   # used only for output file naming
            second_chain= naming_rna,   # used only for output file naming
            run_mode    = "generated",
            input_dir   = rank_path,
            # results_dir auto-set to rank_path/results/ inside run_interface
            pre_split   = {"protein": pro_path, "rna": rna_path},
        )
        logging.info(
            f"[generated] {c_pdb}/{rank_label}  BSA={result['bsa_complex']}  "
            f"has_interface={result['has_interface']}  error={result.get('error')}"
        )
        rank_results[rank_label] = result

    return rank_results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(json_path:      str,
                 truth_dir:      str,
                 gen_dir:        str,
                 out_dir:        str,
                 naccess_path:   str  = "",
                 skip_complex:   bool = False,
                 skip_unbound:   bool = False,
                 skip_generated: bool = False) -> dict:
    """
    Run the full analysis pipeline for all complexes in PRDBv3.json.

    Parameters
    ----------
    json_path      : path to PRDBv3.json
    truth_dir      : root of ALL_PDBs/ directory
    gen_dir        : root of generated_PDBS/ directory
    out_dir        : directory for pipeline outputs
    naccess_path   : optional full path to naccess binary
    skip_complex   : skip Mode A (bound complex)
    skip_unbound   : skip Mode B (unbound structures)
    skip_generated : skip Mode C (generated/docked)

    Returns
    -------
    dict  {C_PDB: {"complex": {...}, "unbound": {...}, "generated": {...}}}
    """
    setup_logging(out_dir)
    start_time = datetime.now()

    logging.info("=" * 70)
    logging.info("FFT-scorer Interface Pipeline  —  START")
    logging.info(f"  JSON     : {json_path}")
    logging.info(f"  truth_dir: {truth_dir}")
    logging.info(f"  gen_dir  : {gen_dir}")
    logging.info(f"  out_dir  : {out_dir}")
    logging.info(f"  Modes    : "
                 f"complex={'OFF' if skip_complex else 'ON'}  "
                 f"unbound={'OFF' if skip_unbound else 'ON'}  "
                 f"generated={'OFF' if skip_generated else 'ON'}")
    logging.info("=" * 70)

    # ── naccess availability ───────────────────────────────────────────────
    patch_naccess_path(naccess_path)
    check_naccess_available()   # warns but does NOT abort — lets pipeline run

    # ── Load JSON ─────────────────────────────────────────────────────────
    with open(json_path, 'r') as f:
        entries = json.load(f)
    logging.info(f"Loaded {len(entries)} entries from {json_path}")

    all_results = {}
    n_total     = len(entries)

    for idx, entry in enumerate(entries, 1):
        c_pdb = entry.get("C_PDB", f"UNKNOWN_{idx}")
        logging.info("")
        logging.info(
            f"── [{idx}/{n_total}] {c_pdb}  "
            f"chains=({entry.get('C_pro_chain')}/{entry.get('C_RNA_chain')})  "
            f"docking={entry.get('Docking_case')} ───"
        )

        rec = {"complex": {}, "unbound": {}, "generated": {}}

        # Mode A — bound complex
        if not skip_complex:
            rec["complex"] = run_complex_mode(entry, truth_dir)

        # Mode B — unbound structures
        if not skip_unbound:
            rec["unbound"] = run_unbound_mode(entry, truth_dir)

        # Mode C — generated / docked
        if not skip_generated:
            rec["generated"] = run_generated_mode(entry, gen_dir)

        all_results[c_pdb] = rec

    # ── Save results JSON ─────────────────────────────────────────────────
    results_json_path = os.path.join(out_dir, "pipeline_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logging.info(f"\nSaved pipeline results → {results_json_path}")

    # ── Comparison tables ─────────────────────────────────────────────────
    logging.info("\nCalling compare_results …")
    try:
        cr.run_comparison(all_results, out_dir)
    except Exception as exc:
        logging.error(
            f"compare_results failed: {exc}\n{traceback.format_exc()}"
        )

    elapsed = datetime.now() - start_time
    logging.info("")
    logging.info("=" * 70)
    logging.info(f"FFT-scorer Interface Pipeline  —  DONE  ({elapsed})")
    logging.info("=" * 70)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="FFT-scorer interface analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json",       required=True,
                   help="Path to PRDBv3.json")
    p.add_argument("--truth_dir",  required=True,
                   help="Root of ALL_PDBs/ directory")
    p.add_argument("--gen_dir",    required=True,
                   help="Root of generated_PDBS/ directory")
    p.add_argument("--out_dir",    required=True,
                   help="Directory for all pipeline outputs")
    p.add_argument("--naccess_path", default="",
                   help="Full path to naccess binary if not on system PATH "
                        "(e.g. /usr/local/bin/naccess).  Only needed when "
                        "naccess is not installed system-wide.")
    p.add_argument("--skip_complex",   action="store_true",
                   help="Skip bound complex analysis (Mode A)")
    p.add_argument("--skip_unbound",   action="store_true",
                   help="Skip unbound structure analysis (Mode B)")
    p.add_argument("--skip_generated", action="store_true",
                   help="Skip generated/docked analysis (Mode C)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        json_path       = args.json,
        truth_dir       = args.truth_dir,
        gen_dir         = args.gen_dir,
        out_dir         = args.out_dir,
        naccess_path    = args.naccess_path,
        skip_complex    = args.skip_complex,
        skip_unbound    = args.skip_unbound,
        skip_generated  = args.skip_generated,
    )