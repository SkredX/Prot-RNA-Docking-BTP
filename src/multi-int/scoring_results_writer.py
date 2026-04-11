#!/usr/bin/python3
# =============================================================================
# scoring_results_writer.py  —  Output generation for scoring results
# =============================================================================
#
# Generates three sets of output files:
#
#   1. Complex-Specific Result Files
#      Location: generated_PDBS/<complex_id>/rank<N>/results/
#      Files: score_details.json (per-rank metrics)
#
#   2. Pipeline-Level Result Files
#      Location: <out_dir>/ (specified by --out_dir)
#      Files: pipeline_results.json (expanded with scoring metrics)
#             residue_comparison.tsv
#             bsa_comparison.tsv
#             scoring_rankings.tsv (main deliverable)
#
# =============================================================================

import os
import json
import logging
from datetime import datetime
from typing import Dict, List


# ---------------------------------------------------------------------------
# Complex-Specific Outputs
# ---------------------------------------------------------------------------

def write_rank_score_details(rank_results_dir: str,
                              complex_id: str,
                              rank_label: str,
                              score_result: Dict) -> str:
    """
    Write score_details.json for a single rank.

    Parameters
    ----------
    rank_results_dir : str
        Path to generated_PDBS/<complex_id>/<rank>/results/
    complex_id : str
        Complex identifier
    rank_label : str
        Rank label (e.g., "rank1")
    score_result : dict
        Output from scoring_engine.score_single_rank()

    Returns
    -------
    str
        Path to written file
    """
    os.makedirs(rank_results_dir, exist_ok=True)

    output_file = os.path.join(rank_results_dir, "score_details.json")

    with open(output_file, 'w') as f:
        json.dump(score_result, f, indent=2, default=str)

    logging.info(f"Wrote rank score details: {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# Pipeline-Level Outputs
# ---------------------------------------------------------------------------

def write_pipeline_results_with_scores(all_results: Dict,
                                       all_scores: Dict,
                                       out_dir: str) -> str:
    """
    Expand pipeline_results.json to include scoring metrics.

    Parameters
    ----------
    all_results : dict
        Original pipeline results {complex_id: {"complex": {...}, ...}}
    all_scores : dict
        Scoring results {complex_id: {rank: score_dict}}
    out_dir : str
        Output directory

    Returns
    -------
    str
        Path to written file
    """
    # Deep copy to avoid modifying original
    expanded_results = {}

    for complex_id, modes in all_results.items():
        expanded_results[complex_id] = {
            "complex": modes.get("complex", {}),
            "unbound": modes.get("unbound", {}),
            "generated": {}
        }

        # Inject scores into generated results
        if complex_id in all_scores:
            for rank_label, gen_result in modes.get("generated", {}).items():
                expanded_results[complex_id]["generated"][rank_label] = {
                    **gen_result,
                    "scores": all_scores[complex_id].get(rank_label, {})
                }
        else:
            expanded_results[complex_id]["generated"] = modes.get("generated", {})

    output_file = os.path.join(out_dir, "pipeline_results_with_scores.json")

    with open(output_file, 'w') as f:
        json.dump(expanded_results, f, indent=2, default=str)

    logging.info(f"Wrote expanded pipeline results: {output_file}")
    return output_file


def write_bsa_comparison_tsv(all_results: Dict, out_dir: str) -> str:
    """
    Write bsa_comparison.tsv: Ground truth vs. generated BSA side-by-side.

    Format:
        Complex_ID | Mode | Rank | BSA_Complex | BSA_Protein | BSA_RNA
        1ASY       | A    |      | 1200.5      | 800.2       | 400.3
        1ASY       | C    | rank1| 1180.2      | 790.1       | 390.1

    Parameters
    ----------
    all_results : dict
        Pipeline results
    out_dir : str
        Output directory

    Returns
    -------
    str
        Path to written file
    """
    output_file = os.path.join(out_dir, "bsa_comparison.tsv")

    with open(output_file, 'w') as f:
        # Header
        f.write("\t".join([
            "Complex_ID",
            "Mode",
            "Rank",
            "BSA_Complex",
            "BSA_Protein",
            "BSA_RNA"
        ]) + "\n")

        # Data
        for complex_id, modes in sorted(all_results.items()):
            # Mode A — complex
            if modes.get("complex"):
                result = modes["complex"]
                f.write("\t".join([
                    complex_id,
                    "A",  # Mode A = complex
                    "",   # No rank
                    str(result.get("bsa_complex", "NA")),
                    str(result.get("bsa_pro", "NA")),
                    str(result.get("bsa_rna", "NA")),
                ]) + "\n")

            # Mode B — unbound (skip for now; no BSA computed)

            # Mode C — generated
            for rank_label, result in sorted(modes.get("generated", {}).items()):
                if result.get("skipped"):
                    continue
                f.write("\t".join([
                    complex_id,
                    "C",  # Mode C = generated
                    rank_label,
                    str(result.get("bsa_complex", "NA")),
                    str(result.get("bsa_pro", "NA")),
                    str(result.get("bsa_rna", "NA")),
                ]) + "\n")

    logging.info(f"Wrote BSA comparison: {output_file}")
    return output_file


def write_residue_comparison_tsv(all_results: Dict, out_dir: str) -> str:
    """
    Write residue_comparison.tsv: Interface residue counts.

    Format:
        Complex_ID | Mode | Rank | Protein_Interface_Residues | RNA_Interface_Residues
        1ASY       | A    |      | 25                         | 18
        1ASY       | C    | rank1| 22                         | 16

    Parameters
    ----------
    all_results : dict
        Pipeline results
    out_dir : str
        Output directory

    Returns
    -------
    str
        Path to written file
    """
    output_file = os.path.join(out_dir, "residue_comparison.tsv")

    def count_interface_residues(int_file: str) -> int:
        """Count lines in .int file (each line = one interface residue)."""
        if not os.path.exists(int_file):
            return 0
        try:
            with open(int_file, 'r') as f:
                return len([l for l in f if l.strip()])
        except:
            return 0

    with open(output_file, 'w') as f:
        # Header
        f.write("\t".join([
            "Complex_ID",
            "Mode",
            "Rank",
            "Protein_Interface_Residues",
            "RNA_Interface_Residues",
        ]) + "\n")

        # Data
        for complex_id, modes in sorted(all_results.items()):
            # Mode A
            if modes.get("complex"):
                result = modes["complex"]
                n_pro = count_interface_residues(result.get("pro_int", ""))
                n_rna = count_interface_residues(result.get("rna_int", ""))
                f.write("\t".join([
                    complex_id,
                    "A",
                    "",
                    str(n_pro),
                    str(n_rna),
                ]) + "\n")

            # Mode C
            for rank_label, result in sorted(modes.get("generated", {}).items()):
                if result.get("skipped"):
                    continue
                n_pro = count_interface_residues(result.get("pro_int", ""))
                n_rna = count_interface_residues(result.get("rna_int", ""))
                f.write("\t".join([
                    complex_id,
                    "C",
                    rank_label,
                    str(n_pro),
                    str(n_rna),
                ]) + "\n")

    logging.info(f"Wrote residue comparison: {output_file}")
    return output_file


def write_scoring_rankings_tsv(all_results: Dict,
                                all_scores: Dict,
                                out_dir: str) -> str:
    """
    Write scoring_rankings.tsv: Main deliverable showing all metrics.

    Format:
        Complex_ID | Rank | Mode_A_BSA | Mode_C_BSA | f_nat | I-RMSD | Δ BSA | Clash | Score | Category
        1ASY       | rank1| 1200.5     | 1180.2     | 0.88  | 1.05   | 20.3  | 0.02  | 0.92  | Near-Native
        1ASY       | rank2| 1200.5     | 800.1      | 0.12  | 8.50   | 400.4 | 0.15  | 0.15  | Incorrect

    Parameters
    ----------
    all_results : dict
        Pipeline results
    all_scores : dict
        Scoring results
    out_dir : str
        Output directory

    Returns
    -------
    str
        Path to written file
    """
    output_file = os.path.join(out_dir, "scoring_rankings.tsv")

    with open(output_file, 'w') as f:
        # Header
        f.write("\t".join([
            "Complex_ID",
            "Rank",
            "Mode_A_BSA",
            "Mode_C_BSA",
            "f_nat",
            "I-RMSD",
            "Δ_BSA",
            "Clash",
            "Interaction_Score",
            "Category",
        ]) + "\n")

        # Data
        for complex_id in sorted(all_results.keys()):
            modes = all_results[complex_id]
            scores = all_scores.get(complex_id, {})

            # Get Mode A (ground truth) BSA
            mode_a_bsa = modes.get("complex", {}).get("bsa_complex", "NA")

            # Mode C (generated) with scores
            for rank_label in sorted(modes.get("generated", {}).keys()):
                gen_result = modes["generated"][rank_label]
                score_result = scores.get(rank_label, {})

                if gen_result.get("skipped") or score_result.get("error"):
                    continue

                mode_c_bsa = gen_result.get("bsa_complex", "NA")

                f.write("\t".join([
                    complex_id,
                    rank_label,
                    str(mode_a_bsa),
                    str(mode_c_bsa),
                    str(score_result.get("f_nat", "NA")),
                    str(score_result.get("i_rmsd", "NA")),
                    str(score_result.get("bsa_delta", "NA")),
                    str(score_result.get("clash_penalty", "NA")),
                    str(score_result.get("interaction_score", "NA")),
                    str(score_result.get("category", "NA")),
                ]) + "\n")

    logging.info(f"Wrote scoring rankings: {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------

def write_scoring_summary(all_scores: Dict, out_dir: str) -> str:
    """
    Write a text summary of scoring statistics.

    Parameters
    ----------
    all_scores : dict
        Scoring results
    out_dir : str
        Output directory

    Returns
    -------
    str
        Path to written file
    """
    output_file = os.path.join(out_dir, "scoring_summary.txt")

    # Aggregate statistics
    n_total = 0
    n_near_native = 0
    n_medium = 0
    n_incorrect = 0
    scores_list = []

    for complex_id, ranks in all_scores.items():
        for rank_label, score_result in ranks.items():
            if score_result.get("error"):
                continue

            n_total += 1
            category = score_result.get("category", "Unknown")
            int_score = score_result.get("interaction_score", 0.0)

            if category == "Near-Native":
                n_near_native += 1
            elif category == "Medium":
                n_medium += 1
            elif category == "Incorrect":
                n_incorrect += 1

            if isinstance(int_score, (int, float)):
                scores_list.append(float(int_score))

    # Compute percentiles
    import statistics
    mean_score = statistics.mean(scores_list) if scores_list else 0.0
    median_score = statistics.median(scores_list) if scores_list else 0.0
    min_score = min(scores_list) if scores_list else 0.0
    max_score = max(scores_list) if scores_list else 0.0

    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FFT-scorer Scoring Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Categorization:\n")
        f.write(f"  Total predictions:   {n_total}\n")
        f.write(f"  Near-Native:         {n_near_native} ({100*n_near_native/max(n_total,1):.1f}%)\n")
        f.write(f"  Medium:              {n_medium} ({100*n_medium/max(n_total,1):.1f}%)\n")
        f.write(f"  Incorrect:           {n_incorrect} ({100*n_incorrect/max(n_total,1):.1f}%)\n\n")

        f.write("Score Distribution:\n")
        f.write(f"  Mean:                {mean_score:.3f}\n")
        f.write(f"  Median:              {median_score:.3f}\n")
        f.write(f"  Min:                 {min_score:.3f}\n")
        f.write(f"  Max:                 {max_score:.3f}\n")

    logging.info(f"Wrote scoring summary: {output_file}")
    return output_file
