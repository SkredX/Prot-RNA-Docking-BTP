#!/usr/bin/python3
# =============================================================================
# run_pipeline.py  —  Orchestrator with integrated scoring
# =============================================================================
#
# OVERVIEW
# --------
# Extended version of the original run_pipeline.py that adds scoring metrics.
# Executes three modes (complex, unbound, generated) and then calls the
# scoring engine to compute interaction metrics for each generated rank.
#
# NEW FEATURES
# ============
# • scoring_engine: Computes f_nat, I-RMSD, Δ BSA, steric clash, composite score
# • scoring_results_writer: Generates TSV and JSON outputs
# • Complex-specific outputs: score_details.json in each rank/results/ folder
# • Pipeline outputs: scoring_rankings.tsv (main deliverable) + supporting files
#
# USAGE
# -----
#   python run_pipeline.py \
#       --json      D:\FFT-scorer\assets\PRDBv3.json \
#       --truth_dir D:\FFT-scorer\assets\ALL_PDBs \
#       --gen_dir   D:\FFT-scorer\src\generated_PDBS \
#       --out_dir   D:\FFT-scorer\src\results \
#       --skip_scoring  (optional: disable scoring if you only want interface analysis)
#
# =============================================================================

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# Locate all helper modules
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import multi_interface as mi
import scoring_engine as se
import scoring_results_writer as srw

# Try to import compare_results if available (backward compatible)
try:
    import compare_results as cr
    HAS_COMPARE_RESULTS = True
except ImportError:
    HAS_COMPARE_RESULTS = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(out_dir: str) -> None:
    """Setup logging to both file and stdout."""
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
    """Strip trailing '*' from PDB ID."""
    return pdb_id.rstrip('*') if isinstance(pdb_id, str) else pdb_id


def expand_chains(chain_str: str) -> list:
    """Convert chain string to list of single-letter IDs."""
    if not chain_str:
        return []
    return list(str(chain_str))


def pdb_folder(truth_dir: str, complex_id: str) -> str:
    """Return ALL_PDBs/<complex_id>/ path."""
    return os.path.join(truth_dir, complex_id)


def rank_folders(gen_dir: str, complex_id: str) -> list:
    """Return sorted list of rank folders for a complex."""
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
    """Standard sentinel for skipped runs."""
    return {
        "skipped": True, "reason": reason,
        "bsa_complex": "NA", "bsa_pro": "NA", "bsa_rna": "NA",
        "pro_int": "NA", "rna_int": "NA", "combined_int": "NA",
        "has_interface": False, "error": None,
    }


def safe_run(label: str, fn, *args, **kwargs) -> dict:
    """Execute fn with exception handling, including sys.exit() from multi_interface."""
    try:
        return fn(*args, **kwargs)
    except SystemExit as exc:
        msg = (f"SystemExit({exc.code}) in {label} — "
               "likely a missing chain or naccess error. Skipping.")
        logging.warning(msg)
        return {
            "bsa_complex": "NA", "bsa_pro": "NA", "bsa_rna": "NA",
            "pro_int": "NA", "rna_int": "NA", "combined_int": "NA",
            "has_interface": False, "error": msg,
        }
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
    """Add naccess directory to PATH."""
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
    """Check if naccess is on PATH."""
    import shutil
    if shutil.which("naccess") is not None:
        logging.info("naccess found on PATH — OK")
        return True
    logging.error(
        "naccess NOT found on PATH.\n"
        "  naccess is a Linux-only binary. Options:\n"
        "    1. Install via conda: conda install -c bioconda naccess\n"
        "    2. Download from: http://www.bioinf.manchester.ac.uk/naccess/\n"
        "    3. Use --naccess_path /path/to/naccess"
    )
    return False


# ---------------------------------------------------------------------------
# Interface Analysis Modes (from original run_pipeline.py)
# ---------------------------------------------------------------------------

def run_complex_mode(entry: dict, truth_dir: str) -> dict:
    """
    Mode A: Analyze bound complex (source-truth).

    Parameters
    ----------
    entry : dict
        One entry from PRDBv3.json
    truth_dir : str
        Root of ALL_PDBs/

    Returns
    -------
    dict keyed by "(pro_chain, rna_chain)" pairs
    """
    c_pdb = entry["C_PDB"]
    c_pro_chain = entry.get("C_pro_chain", "")
    c_rna_chain = entry.get("C_RNA_chain", "")

    pro_chains = expand_chains(c_pro_chain)
    rna_chains = expand_chains(c_rna_chain)

    input_dir = pdb_folder(truth_dir, c_pdb)
    results_dir = os.path.join(input_dir, "complex_results")

    results_dict = {}

    for pc in pro_chains:
        for rc in rna_chains:
            pair_key = pc + rc

            logging.info(
                f"[complex] {c_pdb}  ({pc}/{rc})  "
                f"input={input_dir}  output={results_dir}"
            )

            result = safe_run(
                f"complex/{c_pdb}/{pair_key}",
                mi.run_interface,
                pdb_file=c_pdb,
                first_chain=pc,
                second_chain=rc,
                run_mode="complex",
                input_dir=input_dir,
                results_dir=results_dir,
            )

            logging.info(
                f"[complex] {c_pdb}/{pair_key}  BSA={result['bsa_complex']}  "
                f"has_interface={result['has_interface']}"
            )

            results_dict[pair_key] = result

    return results_dict


def run_unbound_mode(entry: dict, truth_dir: str) -> dict:
    """
    Mode B: Analyze unbound structures (if available).

    Only runs when Docking_case == "UU".

    Parameters
    ----------
    entry : dict
        One entry from PRDBv3.json
    truth_dir : str
        Root of ALL_PDBs/

    Returns
    -------
    dict keyed by "(pro_chain, rna_chain)" pairs
    """
    c_pdb = entry["C_PDB"]
    docking_case = entry.get("Docking_case", "")

    if docking_case != "UU":
        logging.info(f"[unbound] {c_pdb}  skipped — docking_case={docking_case}")
        return {}

    u_pro_pdb = entry.get("U_pro_PDB", "")
    u_rna_pdb = strip_star(entry.get("U_RNA_PDB", ""))
    u_pro_chain = entry.get("U_PRO_chain", "")
    u_rna_chain = entry.get("U_RNA_chain", "")

    input_dir = pdb_folder(truth_dir, c_pdb)
    results_dir = os.path.join(input_dir, "unbound_results")

    pro_path = os.path.join(input_dir, u_pro_pdb + ".pdb")
    rna_path = os.path.join(input_dir, u_rna_pdb + ".pdb")

    if not os.path.exists(pro_path) or not os.path.exists(rna_path):
        msg = f"[unbound] {c_pdb}  skipped — files not found"
        logging.warning(msg)
        return {"unbound": skipped_result(msg)}

    logging.info(
        f"[unbound] {c_pdb}  "
        f"input={input_dir}  output={results_dir}"
    )

    result = safe_run(
        f"unbound/{c_pdb}",
        mi.run_interface,
        pdb_file=c_pdb,
        first_chain=u_pro_chain,
        second_chain=u_rna_chain,
        run_mode="unbound",
        input_dir=input_dir,
        results_dir=results_dir,
        pre_split={"protein": pro_path, "rna": rna_path},
    )

    logging.info(
        f"[unbound] {c_pdb}  BSA={result['bsa_complex']}"
    )

    return {"unbound": result}


def run_generated_mode(entry: dict, gen_dir: str) -> dict:
    """
    Mode C: Analyze all generated docked poses.

    Parameters
    ----------
    entry : dict
        One entry from PRDBv3.json
    gen_dir : str
        Root of generated_PDBS/

    Returns
    -------
    dict keyed by rank label: {"rank1": {...}, "rank2": {...}, ...}
    """
    c_pdb = entry["C_PDB"]
    ranks = rank_folders(gen_dir, c_pdb)

    if not ranks:
        logging.warning(
            f"[generated] {c_pdb}  no rank folders found"
        )
        return {}

    pro_chains = expand_chains(entry.get("C_pro_chain", ""))
    rna_chains = expand_chains(entry.get("C_RNA_chain", ""))
    naming_pro = pro_chains[0] if pro_chains else "P"
    naming_rna = rna_chains[0] if rna_chains else "R"

    rank_results = {}

    for rank_path in ranks:
        rank_label = os.path.basename(rank_path)
        pro_path = os.path.join(rank_path, "protein.pdb")
        rna_path = os.path.join(rank_path, "rna.pdb")

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
            pdb_file=c_pdb,
            first_chain=naming_pro,
            second_chain=naming_rna,
            run_mode="generated",
            input_dir=rank_path,
            results_dir=os.path.join(rank_path, "results"),
            pre_split={"protein": pro_path, "rna": rna_path},
        )

        logging.info(
            f"[generated] {c_pdb}/{rank_label}  BSA={result['bsa_complex']}  "
            f"has_interface={result['has_interface']}"
        )

        rank_results[rank_label] = result

    return rank_results


# ---------------------------------------------------------------------------
# NEW: Scoring Integration
# ---------------------------------------------------------------------------

def score_generated_poses(complex_id: str,
                          truth_result: dict,
                          gen_results: dict,
                          gen_base_dir: str,
                          truth_dir: str = "") -> dict:
    """
    Score all generated poses against ground truth.

    Parameters
    ----------
    complex_id   : str  — e.g. "1ASY"
    truth_result : dict — Mode A interface analysis result
    gen_results  : dict — {rank_label: interface_result} from Mode C
    gen_base_dir : str  — root of generated_PDBS/
    truth_dir    : str  — root of ALL_PDBs/ (used to resolve truth PDB path)

    Returns
    -------
    dict  {rank_label: score_dict}
    """
    scores = {}

    if truth_result.get("error") or truth_result.get("skipped"):
        logging.warning(
            f"[scoring] {complex_id}  skipped — ground truth analysis failed"
        )
        return {}

    truth_combined_int = truth_result.get("combined_int", "") or ""

    # Resolve ground truth PDB: try result dict first, then several canonical paths
    truth_pdb = truth_result.get("combined_pdb") or truth_result.get("pdb_file") or ""
    if not truth_pdb or not os.path.exists(truth_pdb):
        # Try common filename patterns inside ALL_PDBs/<complex_id>/
        candidates = [
            os.path.join(truth_dir, complex_id, f"{complex_id}.pdb"),
            os.path.join(truth_dir, complex_id, f"{complex_id}_complex.pdb"),
            os.path.join(truth_dir, complex_id, "complex.pdb"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                truth_pdb = cand
                logging.info(f"[scoring] {complex_id}  resolved truth PDB → {cand!r}")
                break
        else:
            logging.warning(
                f"[scoring] {complex_id}  cannot resolve truth PDB "
                f"(tried result dict and {candidates}) — scoring will use "
                "distance-based fallback for interface detection"
            )
            truth_pdb = ""   # score_single_rank will handle missing file

    # .int file existence is optional — scoring_engine falls back to distance
    if truth_combined_int and not os.path.exists(truth_combined_int):
        logging.warning(
            f"[scoring] {complex_id}  truth .int not found: {truth_combined_int!r} "
            "— will use distance-based interface detection"
        )
        truth_combined_int = ""

    for rank_label, gen_result in gen_results.items():
        if gen_result.get("error") or gen_result.get("skipped"):
            logging.warning(
                f"[scoring] {complex_id}/{rank_label}  "
                f"skipped due to interface analysis error"
            )
            scores[rank_label] = {
                "error": gen_result.get("error", "Analysis failed"),
                "interaction_score": 0.0,
                "category": "Skipped"
            }
            continue

        # Resolve file paths
        rank_dir    = os.path.join(gen_base_dir, complex_id, rank_label)
        results_dir = os.path.join(rank_dir, "results")

        gen_pdb      = os.path.join(rank_dir, "combined.pdb")
        gen_int_file = gen_result.get("combined_int", "") or ""
        protein_pdb  = os.path.join(rank_dir, "protein.pdb")
        rna_pdb      = os.path.join(rank_dir, "rna.pdb")

        # Build combined PDB if it doesn't exist
        if not os.path.exists(gen_pdb):
            try:
                with open(gen_pdb, 'w') as combined:
                    with open(protein_pdb) as pro:
                        combined.write(pro.read())
                    with open(rna_pdb) as rna:
                        combined.write(rna.read())
            except Exception as e:
                logging.error(f"Failed to create combined PDB: {e}")
                scores[rank_label] = {
                    "error": f"Failed to create combined PDB: {e}",
                    "interaction_score": 0.0,
                    "category": "Error"
                }
                continue

        # Call scoring engine
        logging.info(f"[scoring] {complex_id}/{rank_label}  computing metrics…")
        score_result = se.score_single_rank(
            complex_id=complex_id,
            truth_pdb=truth_pdb,
            truth_bsa=truth_result.get("bsa_complex", "NA"),
            gen_pdb=gen_pdb,
            gen_bsa=gen_result.get("bsa_complex", "NA"),
            truth_int_file=truth_combined_int,
            gen_int_file=gen_int_file,
            protein_pdb=protein_pdb,
            rna_pdb=rna_pdb,
            rank_label=rank_label,
        )

        # Write rank-specific score file
        try:
            os.makedirs(results_dir, exist_ok=True)
            srw.write_rank_score_details(results_dir, complex_id, rank_label, score_result)
        except Exception as e:
            logging.error(f"Failed to write rank score details: {e}")

        logging.info(
            f"[scoring] {complex_id}/{rank_label}  "
            f"score={score_result.get('interaction_score', 'NA')}  "
            f"category={score_result.get('category', 'NA')}"
        )

        scores[rank_label] = score_result

    return scores


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(json_path: str,
                 truth_dir: str,
                 gen_dir: str,
                 out_dir: str,
                 naccess_path: str = "",
                 skip_complex: bool = False,
                 skip_unbound: bool = False,
                 skip_generated: bool = False,
                 skip_scoring: bool = False,
                 pdb_filter: str = "") -> dict:
    """
    Run the full pipeline including scoring.

    Parameters
    ----------
    json_path      : path to PRDBv3.json
    truth_dir      : root of ALL_PDBs/
    gen_dir        : root of generated_PDBS/
    out_dir        : output directory
    naccess_path   : optional path to naccess binary
    skip_complex   : skip Mode A
    skip_unbound   : skip Mode B
    skip_generated : skip Mode C
    skip_scoring   : skip scoring (new option)

    Returns
    -------
    dict : {complex_id: {"complex": {...}, "unbound": {...}, "generated": {...}, "scores": {...}}}
    """
    setup_logging(out_dir)
    start_time = datetime.now()

    logging.info("=" * 70)
    logging.info("FFT-scorer Pipeline with Scoring  —  START")
    logging.info(f"  JSON     : {json_path}")
    logging.info(f"  truth_dir: {truth_dir}")
    logging.info(f"  gen_dir  : {gen_dir}")
    logging.info(f"  out_dir  : {out_dir}")
    logging.info(f"  Modes    : "
                 f"complex={'OFF' if skip_complex else 'ON'}  "
                 f"unbound={'OFF' if skip_unbound else 'ON'}  "
                 f"generated={'OFF' if skip_generated else 'ON'}")
    logging.info(f"  Scoring  : {'OFF' if skip_scoring else 'ON'}")
    logging.info("=" * 70)

    patch_naccess_path(naccess_path)
    check_naccess_available()

    with open(json_path, 'r') as f:
        entries = json.load(f)
    logging.info(f"Loaded {len(entries)} entries from {json_path}")

    # Apply optional PDB filter
    if pdb_filter:
        allowed = {p.strip().upper() for p in pdb_filter.split(",") if p.strip()}
        entries = [e for e in entries if str(e.get("C_PDB", "")).upper() in allowed]
        logging.info(
            f"--pdb_filter applied: processing {len(entries)} entr"
            f"{'y' if len(entries)==1 else 'ies'} "
            f"({', '.join(sorted(allowed))})"
        )
        if not entries:
            logging.error(
                f"No entries match --pdb_filter={pdb_filter!r}. "
                "Check that the PDB IDs exist in the JSON file."
            )
            return {}

    all_results = {}
    all_scores = {}
    n_total = len(entries)

    for idx, entry in enumerate(entries, 1):
        c_pdb = entry.get("C_PDB", f"UNKNOWN_{idx}")
        logging.info("")
        logging.info(
            f"── [{idx}/{n_total}] {c_pdb}  "
            f"chains=({entry.get('C_pro_chain')}/{entry.get('C_RNA_chain')})  "
            f"docking={entry.get('Docking_case')} ───"
        )

        rec = {"complex": {}, "unbound": {}, "generated": {}}

        # Mode A
        if not skip_complex:
            rec["complex"] = run_complex_mode(entry, truth_dir)

        # Mode B
        if not skip_unbound:
            rec["unbound"] = run_unbound_mode(entry, truth_dir)

        # Mode C
        if not skip_generated:
            rec["generated"] = run_generated_mode(entry, gen_dir)

        all_results[c_pdb] = rec

        # NEW: Scoring
        if not skip_scoring and not skip_generated and rec.get("complex") and rec.get("generated"):
            # Get first complex result (Mode A) for ground truth
            truth_result = next(iter(rec["complex"].values()), {})

            scores = score_generated_poses(
                complex_id=c_pdb,
                truth_result=truth_result,
                gen_results=rec["generated"],
                gen_base_dir=gen_dir,
                truth_dir=truth_dir,
            )
            all_scores[c_pdb] = scores

    # Save pipeline results JSON
    results_json_path = os.path.join(out_dir, "pipeline_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logging.info(f"\nSaved pipeline results → {results_json_path}")

    # NEW: Write scoring outputs
    if not skip_scoring and all_scores:
        logging.info("\nGenerating scoring outputs…")
        try:
            srw.write_pipeline_results_with_scores(all_results, all_scores, out_dir)
            srw.write_bsa_comparison_tsv(all_results, out_dir)
            srw.write_residue_comparison_tsv(all_results, out_dir)
            srw.write_scoring_rankings_tsv(all_results, all_scores, out_dir)
            srw.write_scoring_summary(all_scores, out_dir)
        except Exception as exc:
            logging.error(
                f"Scoring output generation failed: {exc}\n{traceback.format_exc()}"
            )

    # Optional: compare_results (if available)
    if HAS_COMPARE_RESULTS:
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
    logging.info(f"FFT-scorer Pipeline  —  DONE  ({elapsed})")
    logging.info("=" * 70)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="FFT-scorer pipeline with scoring metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json", required=True,
                   help="Path to PRDBv3.json")
    p.add_argument("--truth_dir", required=True,
                   help="Root of ALL_PDBs/ directory")
    p.add_argument("--gen_dir", required=True,
                   help="Root of generated_PDBS/ directory")
    p.add_argument("--out_dir", required=True,
                   help="Directory for all pipeline outputs")
    p.add_argument("--naccess_path", default="",
                   help="Full path to naccess binary")
    p.add_argument("--skip_complex", action="store_true",
                   help="Skip Mode A (bound complex)")
    p.add_argument("--skip_unbound", action="store_true",
                   help="Skip Mode B (unbound)")
    p.add_argument("--skip_generated", action="store_true",
                   help="Skip Mode C (generated)")
    p.add_argument("--skip_scoring", action="store_true",
                   help="Skip scoring (only compute BSA/interfaces)")
    p.add_argument("--pdb_filter", default="",
                   help="Comma-separated list of C_PDB IDs to process "
                        "(e.g. '1ASY,1AV6'). If omitted, all entries in "
                        "the JSON are processed.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        json_path=args.json,
        truth_dir=args.truth_dir,
        gen_dir=args.gen_dir,
        out_dir=args.out_dir,
        naccess_path=args.naccess_path,
        skip_complex=args.skip_complex,
        skip_unbound=args.skip_unbound,
        skip_generated=args.skip_generated,
        skip_scoring=args.skip_scoring,
        pdb_filter=args.pdb_filter,
    )