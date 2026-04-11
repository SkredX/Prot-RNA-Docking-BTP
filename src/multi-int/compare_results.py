#!/usr/bin/python3
# =============================================================================
# compare_results.py  —  BSA + residue-level comparison engine
# =============================================================================
#
# OVERVIEW
# --------
# Consumes the master results dict produced by run_pipeline.py and generates
# three TSV output files:
#
#   bsa_comparison.tsv
#       BSA values for every (complex, mode, chain pair, rank) combination.
#       Columns:
#           PDB_ID | Mode | ChainPair | Rank |
#           BSA_Complex | BSA_Pro | BSA_RNA |
#           Truth_BSA | Delta_BSA | Delta_Pct | Has_Interface
#
#   residue_comparison.tsv
#       Interface residue overlap of each generated rank vs source truth.
#       Truth residues are taken from the matching chain pair of the bound
#       complex (Mode A).  If multiple chain pairs exist, each rank is compared
#       against the lexicographically first non-error pair.
#       Columns:
#           PDB_ID | ChainPair | Rank |
#           Jaccard | Precision | Recall |
#           Shared | Truth_Only | Rank_Only | Total_Truth | Total_Rank
#
#   summary.tsv
#       One row per complex — best rank by BSA delta and by Jaccard.
#       Columns:
#           PDB_ID | Truth_BSA | Unbound_BSA |
#           Best_Rank_by_BSA | Best_BSA_Delta | Best_BSA_Delta_Pct |
#           Best_Rank_by_Jaccard | Best_Jaccard | N_Ranks_Run
#
# RESULT DICT STRUCTURE (from run_pipeline.py)
# ---------------------------------------------
#   {
#     "1ASY": {
#       "complex":   {"AR": {run_interface result}, "BR": {...}, ...},
#       "unbound":   {"AA": {run_interface result}} | {"_skipped": {...}},
#       "generated": {"rank1": {run_interface result}, "rank2": {...}, ...}
#     },
#     ...
#   }
#
# PUBLIC API
# ----------
#   run_comparison(all_results, out_dir)   — called by run_pipeline automatically
#   parse_int_residues(int_path)           — exposed for testing / ad-hoc use
# =============================================================================

import os
import csv
import json
import logging


# ---------------------------------------------------------------------------
# Residue parsing
# ---------------------------------------------------------------------------

def parse_int_residues(int_path: str) -> set:
    """
    Parse a .int file and return a set of "<chain>:<resnum>" residue identifiers.

    Column positions (fixed-width PDB-derived format written by
    generate_interface_atomfile):
        [20:22]  chain identifier
        [22:26]  residue sequence number

    Returns an empty set if the file does not exist, is empty, or cannot be read.
    """
    residues = set()
    if not int_path or int_path == "NA" or not os.path.isfile(int_path):
        return residues
    try:
        with open(int_path) as f:
            for line in f:
                line = line.rstrip('\n')
                if len(line) < 26:
                    continue
                chain  = line[20:22].strip()
                resnum = line[22:26].strip()
                if chain and resnum:
                    residues.add(f"{chain}:{resnum}")
    except Exception as exc:
        logging.warning(f"Could not parse residues from {int_path}: {exc}")
    return residues


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|.  Returns 0.0 if both empty."""
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def precision_recall(truth: set, predicted: set):
    """
    Precision = |truth ∩ predicted| / |predicted|
    Recall    = |truth ∩ predicted| / |truth|
    Returns (0.0, 0.0) when a denominator is zero.
    """
    shared = len(truth & predicted)
    prec   = shared / len(predicted) if predicted else 0.0
    rec    = shared / len(truth)      if truth     else 0.0
    return prec, rec


# ---------------------------------------------------------------------------
# BSA helpers
# ---------------------------------------------------------------------------

def safe_float(value, fallback="NA"):
    """Return float(value) if possible, else fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def bsa_delta(truth_bsa, rank_bsa):
    """
    Compute (rank_bsa − truth_bsa) and its percentage.
    Returns ("NA", "NA") when either input is unavailable.
    """
    t = safe_float(truth_bsa)
    r = safe_float(rank_bsa)
    if t == "NA" or r == "NA":
        return "NA", "NA"
    delta     = round(r - t, 2)
    delta_pct = round(delta / t * 100, 2) if t != 0 else "NA"
    return delta, delta_pct


# ---------------------------------------------------------------------------
# Result dict helpers
# ---------------------------------------------------------------------------

def is_valid_result(result: dict) -> bool:
    """
    Return True if the result dict represents a real run_interface() outcome
    (not a skip sentinel or error placeholder).
    """
    if not result or not isinstance(result, dict):
        return False
    if result.get("skipped"):
        return False
    return True


def best_complex_result(pair_results: dict) -> tuple:
    """
    From a complex/unbound pair_results dict, return (pair_key, result) for
    the first valid (non-skipped, non-error) pair in sorted order.
    Returns (None, {}) if nothing valid is found.
    """
    for key in sorted(pair_results.keys()):
        if key.startswith("_"):         # sentinel keys like "_skipped"
            continue
        result = pair_results[key]
        if is_valid_result(result) and result.get("error") is None:
            return key, result
    return None, {}


# ---------------------------------------------------------------------------
# TSV helpers
# ---------------------------------------------------------------------------

def _open_tsv(path: str, fieldnames: list):
    """Open a TSV writer.  Returns (file_handle, DictWriter)."""
    fh = open(path, 'w', newline='')
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter='\t',
                            extrasaction='ignore')
    writer.writeheader()
    return fh, writer


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def run_comparison(all_results: dict, out_dir: str) -> None:
    """
    Generate bsa_comparison.tsv, residue_comparison.tsv, and summary.tsv.

    Parameters
    ----------
    all_results : dict
        Master results dict as returned by run_pipeline.run_pipeline().
    out_dir : str
        Directory where the three TSV files are written.
    """
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"[compare] Writing outputs to {out_dir}")

    bsa_path  = os.path.join(out_dir, "bsa_comparison.tsv")
    res_path  = os.path.join(out_dir, "residue_comparison.tsv")
    summ_path = os.path.join(out_dir, "summary.tsv")

    bsa_fields = [
        "PDB_ID", "Mode", "ChainPair", "Rank",
        "BSA_Complex", "BSA_Pro", "BSA_RNA",
        "Truth_BSA", "Delta_BSA", "Delta_Pct",
        "Has_Interface",
    ]
    res_fields = [
        "PDB_ID", "ChainPair", "Rank",
        "Jaccard", "Precision", "Recall",
        "Shared", "Truth_Only", "Rank_Only",
        "Total_Truth", "Total_Rank",
    ]
    summ_fields = [
        "PDB_ID",
        "Truth_BSA", "Unbound_BSA",
        "Best_Rank_by_BSA", "Best_BSA_Delta", "Best_BSA_Delta_Pct",
        "Best_Rank_by_Jaccard", "Best_Jaccard",
        "N_Ranks_Run",
    ]

    bsa_fh,  bsa_writer  = _open_tsv(bsa_path,  bsa_fields)
    res_fh,  res_writer  = _open_tsv(res_path,  res_fields)
    summ_fh, summ_writer = _open_tsv(summ_path, summ_fields)

    try:
        for pdb_id, rec in all_results.items():

            complex_pairs   = rec.get("complex",   {})
            unbound_pairs   = rec.get("unbound",   {})
            generated_ranks = rec.get("generated", {})

            # ── Pick the canonical "truth" result ─────────────────────────
            # Use the first valid chain pair from the bound complex.
            # For most entries this is the only pair (e.g. "AR").
            # For multi-chain entries (e.g. "ABKM" × "E") we take the first
            # valid result; BSA and residues for all pairs are written to the
            # full tables below.
            truth_pair_key, truth_result = best_complex_result(complex_pairs)
            truth_bsa      = safe_float(truth_result.get("bsa_complex"))
            truth_int      = truth_result.get("combined_int", "")
            truth_residues = parse_int_residues(truth_int)

            # ── Unbound BSA (first valid pair) ────────────────────────────
            _, unbound_result = best_complex_result(unbound_pairs)
            unbound_bsa       = safe_float(unbound_result.get("bsa_complex"))

            # ── Write BSA rows for "complex" mode (all chain pairs) ───────
            for pair_key, result in complex_pairs.items():
                if pair_key.startswith("_") or not is_valid_result(result):
                    continue
                bsa_writer.writerow({
                    "PDB_ID":        pdb_id,
                    "Mode":          "complex",
                    "ChainPair":     pair_key,
                    "Rank":          "—",
                    "BSA_Complex":   safe_float(result.get("bsa_complex")),
                    "BSA_Pro":       safe_float(result.get("bsa_pro")),
                    "BSA_RNA":       safe_float(result.get("bsa_rna")),
                    "Truth_BSA":     "—",
                    "Delta_BSA":     "—",
                    "Delta_Pct":     "—",
                    "Has_Interface": result.get("has_interface", False),
                })

            # ── Write BSA rows for "unbound" mode ─────────────────────────
            for pair_key, result in unbound_pairs.items():
                if pair_key.startswith("_") or not is_valid_result(result):
                    continue
                ub_bsa = safe_float(result.get("bsa_complex"))
                delta, delta_pct = bsa_delta(truth_bsa, ub_bsa)
                bsa_writer.writerow({
                    "PDB_ID":        pdb_id,
                    "Mode":          "unbound",
                    "ChainPair":     pair_key,
                    "Rank":          "—",
                    "BSA_Complex":   ub_bsa,
                    "BSA_Pro":       safe_float(result.get("bsa_pro")),
                    "BSA_RNA":       safe_float(result.get("bsa_rna")),
                    "Truth_BSA":     truth_bsa,
                    "Delta_BSA":     delta,
                    "Delta_Pct":     delta_pct,
                    "Has_Interface": result.get("has_interface", False),
                })

            # ── Per-rank rows + residue comparison ────────────────────────
            best_rank_bsa     = None
            best_delta_abs    = None
            best_delta        = "NA"
            best_delta_pct    = "NA"
            best_rank_jaccard = None
            best_jaccard_val  = -1.0
            n_ranks_run       = 0

            for rank_label, result in generated_ranks.items():
                if not is_valid_result(result):
                    continue
                n_ranks_run += 1

                rank_bsa          = safe_float(result.get("bsa_complex"))
                delta, delta_pct  = bsa_delta(truth_bsa, rank_bsa)

                # Determine chain pair label for generated mode
                # run_interface() uses the first chain letter of each set as
                # the naming stem, e.g. "1ASY_A.int" / "1ASY_R.int"
                rank_chain_pair = (
                    truth_pair_key if truth_pair_key else "generated"
                )

                # BSA row for this rank
                bsa_writer.writerow({
                    "PDB_ID":        pdb_id,
                    "Mode":          "generated",
                    "ChainPair":     rank_chain_pair,
                    "Rank":          rank_label,
                    "BSA_Complex":   rank_bsa,
                    "BSA_Pro":       safe_float(result.get("bsa_pro")),
                    "BSA_RNA":       safe_float(result.get("bsa_rna")),
                    "Truth_BSA":     truth_bsa,
                    "Delta_BSA":     delta,
                    "Delta_Pct":     delta_pct,
                    "Has_Interface": result.get("has_interface", False),
                })

                # Residue-level comparison vs truth
                rank_int      = result.get("combined_int", "")
                rank_residues = parse_int_residues(rank_int)

                jac           = jaccard(truth_residues, rank_residues)
                prec, rec     = precision_recall(truth_residues, rank_residues)
                shared        = len(truth_residues & rank_residues)
                truth_only    = len(truth_residues - rank_residues)
                rank_only     = len(rank_residues  - truth_residues)

                res_writer.writerow({
                    "PDB_ID":      pdb_id,
                    "ChainPair":   rank_chain_pair,
                    "Rank":        rank_label,
                    "Jaccard":     round(jac,  4),
                    "Precision":   round(prec, 4),
                    "Recall":      round(rec,  4),
                    "Shared":      shared,
                    "Truth_Only":  truth_only,
                    "Rank_Only":   rank_only,
                    "Total_Truth": len(truth_residues),
                    "Total_Rank":  len(rank_residues),
                })

                # Track best rank by BSA (absolute delta closest to zero)
                if delta != "NA":
                    abs_delta = abs(delta)
                    if best_delta_abs is None or abs_delta < best_delta_abs:
                        best_delta_abs    = abs_delta
                        best_delta        = delta
                        best_delta_pct    = delta_pct
                        best_rank_bsa     = rank_label

                # Track best rank by Jaccard
                if jac > best_jaccard_val:
                    best_jaccard_val  = jac
                    best_rank_jaccard = rank_label

            # ── Summary row ───────────────────────────────────────────────
            summ_writer.writerow({
                "PDB_ID":               pdb_id,
                "Truth_BSA":            truth_bsa,
                "Unbound_BSA":          unbound_bsa,
                "Best_Rank_by_BSA":     best_rank_bsa     or "NA",
                "Best_BSA_Delta":       best_delta,
                "Best_BSA_Delta_Pct":   best_delta_pct,
                "Best_Rank_by_Jaccard": best_rank_jaccard or "NA",
                "Best_Jaccard":         round(best_jaccard_val, 4)
                                        if best_jaccard_val >= 0 else "NA",
                "N_Ranks_Run":          n_ranks_run,
            })

            logging.info(
                f"[compare] {pdb_id}  truth_BSA={truth_bsa}  "
                f"unbound_BSA={unbound_bsa}  "
                f"best_BSA_rank={best_rank_bsa}(Δ={best_delta})  "
                f"best_J_rank={best_rank_jaccard}"
                f"(J={round(best_jaccard_val, 3) if best_jaccard_val >= 0 else 'NA'})"
            )

    finally:
        bsa_fh.close()
        res_fh.close()
        summ_fh.close()

    logging.info(f"[compare] bsa_comparison.tsv     → {bsa_path}")
    logging.info(f"[compare] residue_comparison.tsv  → {res_path}")
    logging.info(f"[compare] summary.tsv             → {summ_path}")


# ---------------------------------------------------------------------------
# Standalone entry point — re-run from a saved pipeline_results.json
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Re-run comparison from a saved pipeline_results.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_json", required=True,
                   help="Path to pipeline_results.json from run_pipeline.py")
    p.add_argument("--out_dir",      required=True,
                   help="Directory where TSV files will be written")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.results_json) as f:
        all_results = json.load(f)

    run_comparison(all_results, args.out_dir)