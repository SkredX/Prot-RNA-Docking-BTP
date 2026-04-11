"""
run.py — Interactive Entry Point for the Protein-RNA FFT Docking Pipeline
==========================================================================
This is the single script you run to execute the full docking pipeline.

Usage examples
--------------
# Dock a single complex (no visualizations):
    python run.py --complex 1ASY

# Dock a single complex AND show all visualizations:
    python run.py --complex 1ASY --viz

# Choose which visualizations to show:
    python run.py --complex 1ASY --viz structure grid rotations

# Dock several specific complexes:
    python run.py --complex 1ASY 1AV6 1B23

# Dock every UU case in the dataset:
    python run.py --all

# Full control over parameters:
    python run.py --complex 1ASY --step 15.0 --resolution 1.0 --top_n 10

# Override the default data paths (optional — defaults point to your BTP folder):
    python run.py --all --json "D:\\BTP Files\\PRDBv3.0\\PRDBv3_info.json" \\
                        --pdb_root "D:\\BTP Files\\PRDBv3.0"

Visualization flags
-------------------
--viz                     Show ALL three visualization types (structure + grid + rotations)
--viz structure           Phase 1: 3-D scatter of protein & RNA atom positions
--viz grid                Phase 2: 3-D voxel grid (surface=red, interior=blue)
--viz rotations           Phase 3: SO(3) rotation-axis distribution + angle histogram
--viz structure grid      Any combination of the above is valid

Notes on --viz when processing multiple complexes
--------------------------------------------------
  * "structure" and "grid" open one browser tab per complex — be mindful
    when using --all on a large dataset.
  * "rotations" opens only once (the SO3 set is shared across all complexes).

Pipeline stages executed for every selected complex
----------------------------------------------------
  Phase 1  (phase1.py)   — parse PDB files, detect protein / RNA chains
  Phase 2  (phase2.py)   — voxelise structures into 3-D shape grids
  Phase 3  (phase3.py)   — generate uniform SO(3) rotation set   [shared]
  Phase 4  (phase4.py)   — GPU-accelerated FFT cross-correlation search
  Phase 5  (phase5.py)   — write docked PDB files + RMSD benchmarking

Output
------
  results.pkl            — raw docking results for all processed complexes
  generated_PDBS/
      <COMPLEX_ID>/
          rank1/  protein.pdb  rna.pdb
          rank2/  protein.pdb  rna.pdb
          ...
"""

import os
import sys
import pickle
import argparse
import time
from typing import List, Dict, Set

# ── Pipeline phase imports ─────────────────────────────────────────────────
from phase1 import load_cases, print_summary, validate_case, DockingCase, visualize_structure
from phase2 import GridBuilder, build_grids_for_case, visualize_grid
from phase3 import SO3Sampler, visualize_rotation_axes, visualize_rotation_angles
from phase4 import FFTDocker, DockingResult
from phase5 import run_phase5, BenchmarkResult


# ══════════════════════════════════════════════════════════════════════════
# Defaults — point directly at your BTP data folder
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_JSON     = "../assets/PRDBv3.json"
DEFAULT_PDB_ROOT = "../assets/ALL_PDBs"
DEFAULT_OUTPUT   = "generated_PDBS"
DEFAULT_RESULTS  = "results.pkl"

# All valid visualization type names the user can pass to --viz
VIZ_CHOICES = {"structure", "grid", "rotations"}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _banner(text: str, width: int = 70, char: str = "═") -> None:
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _load_and_filter_cases(
    json_path: str,
    pdb_root:  str,
    requested: List[str],   # empty list means --all
) -> List[DockingCase]:
    """
    Load all UU cases from JSON, then optionally filter to the IDs the user
    requested. Prints a clear error for any complex ID that cannot be found.
    """
    _banner("Phase 1 — Loading & parsing PDB structures")
    print(f"  JSON     : {json_path}")
    print(f"  PDB root : {pdb_root}\n")

    if not os.path.isfile(json_path):
        print(f"[ERROR] JSON file not found: {json_path}")
        print("        Check the --json argument or verify the path exists.")
        sys.exit(1)

    cases, skipped = load_cases(json_path, pdb_root)

    if not cases:
        print("[ERROR] No UU docking cases could be loaded.")
        print(f"        {len(skipped)} case(s) were skipped — check paths and file names.")
        sys.exit(1)

    print_summary(cases, skipped)

    # Build a lookup map (case ID is always stored upper-case after load_cases)
    case_map: Dict[str, DockingCase] = {c.complex_id.upper(): c for c in cases}

    if not requested:
        # --all: use everything that loaded successfully
        selected = cases
    else:
        selected = []
        for raw_id in requested:
            uid = raw_id.strip().upper()
            if uid in case_map:
                selected.append(case_map[uid])
            else:
                print(f"  [WARNING] Complex '{raw_id}' not found in loaded UU cases — skipping.")
                close = [k for k in case_map if k.startswith(uid[:3])]
                if close:
                    print(f"            Did you mean one of: {close[:6]}?")

    if not selected:
        print("\n[ERROR] None of the requested complex IDs could be matched.")
        print("        Available IDs (first 20):", [c.complex_id for c in cases[:20]])
        sys.exit(1)

    # Validate each selected case and warn (but don't abort) on issues
    for case in selected:
        warnings = validate_case(case)
        for w in warnings:
            print(f"  ⚠  {w}")

    mode = "ALL" if not requested else f"{len(selected)} specific complex(es)"
    print(f"\n  → Running pipeline on {mode}: "
          f"{[c.complex_id for c in selected]}\n")

    return selected


# ══════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ══════════════════════════════════════════════════════════════════════════

def _viz_structure(case: DockingCase) -> None:
    """
    Phase 1 visualization — 3-D scatter of every atom coloured by chain type.
    Shows the bound complex (protein + RNA together as they appear in the
    crystal structure).  Opens one browser tab per complex.
    """
    print(f"  [viz] Opening Phase 1 structure plot for {case.complex_id} "
          f"in your browser …")
    visualize_structure(
        case.complex_struct,
        title=f"Complex {case.complex_id} — bound structure (Phase 1)",
    )


def _viz_grid(case: DockingCase) -> None:
    """
    Phase 2 visualization — voxel grids for the protein AND the RNA.
    Surface voxels (+1) are shown in red, interior voxels (-15) in blue.
    Opens two browser tabs: one for protein, one for RNA.
    """
    print(f"  [viz] Building voxel grids for {case.complex_id} …")
    builder = GridBuilder()  # uses default resolution / padding
    protein_grid, rna_grid = build_grids_for_case(case, builder)

    print(f"  [viz] Opening Phase 2 protein voxel grid plot …")
    visualize_grid(protein_grid)

    print(f"  [viz] Opening Phase 2 RNA voxel grid plot …")
    visualize_grid(rna_grid)


def _viz_rotations(sampler: SO3Sampler) -> None:
    """
    Phase 3 visualization — SO(3) rotation-axis distribution on the unit
    sphere plus a histogram of rotation angles.
    Opens two browser tabs (axes + histogram).  Called only once because
    the rotation set is shared across all complexes.
    """
    print("  [viz] Opening Phase 3 rotation-axis distribution plot …")
    visualize_rotation_axes(sampler.rotations)

    print("  [viz] Opening Phase 3 rotation-angle histogram …")
    visualize_rotation_angles(sampler.rotations)


# ══════════════════════════════════════════════════════════════════════════
# Main pipeline runner
# ══════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace) -> None:

    pipeline_start = time.time()

    # Normalise the set of requested visualization types.
    # args.viz is either None (flag absent) or a list of strings.
    viz: Set[str] = set(args.viz) if args.viz is not None else set()

    # ── Phase 1: load & filter cases ────────────────────────────────────
    cases = _load_and_filter_cases(
        json_path  = args.json,
        pdb_root   = args.pdb_root,
        requested  = args.complex,
    )

    # ── Phase 3: build SO(3) rotation set (shared across all complexes) ──
    _banner("Phase 3 — Building SO(3) rotation set (shared across all complexes)")
    docker = FFTDocker(angular_step=args.step, resolution=args.resolution)

    # Phase 3 visualization — opened once here, before the per-complex loop,
    # because the rotation set is the same for every complex.
    if "rotations" in viz:
        _banner("Phase 3 Visualization — SO(3) rotation sampling", char="─")
        _viz_rotations(docker.sampler)

    # ── Phases 4 + 5: process each complex in sequence ──────────────────
    all_results:    Dict[str, List[DockingResult]] = {}
    all_benchmarks: List[BenchmarkResult]          = []

    for idx, case in enumerate(cases, start=1):
        _banner(
            f"Complex {idx}/{len(cases)}: {case.complex_id}  "
            f"— Phases 2 & 4 (grid + FFT docking)",
            char="─",
        )

        # Phase 1 visualization — atom scatter for this complex
        if "structure" in viz:
            _banner(f"Phase 1 Visualization — atom scatter for {case.complex_id}", char="─")
            _viz_structure(case)

        # Phase 2 visualization — voxel grids for this complex
        # Note: this builds the grids a second time purely for display.
        # The actual docking grids are built inside phase4/FFTDocker.dock().
        if "grid" in viz:
            _banner(f"Phase 2 Visualization — voxel grids for {case.complex_id}", char="─")
            _viz_grid(case)

        # Phase 4: FFT docking (the heavy computation)
        docking_results = docker.dock(case)
        all_results[case.complex_id] = docking_results

        # Phase 5: export docked PDB files + RMSD benchmarking
        benchmarks = run_phase5(
            case            = case,
            docking_results = docking_results,
            output_root     = args.output,
            top_n           = args.top_n,
        )
        all_benchmarks.extend(benchmarks)

    # ── Save raw results pickle ──────────────────────────────────────────
    with open(args.results, "wb") as fh:
        pickle.dump(all_results, fh)
    print(f"\n  Results pickle saved → {args.results}")

    # ── Global summary table ─────────────────────────────────────────────
    _banner("Global Benchmark Summary")
    print(f"  {'Complex':<12} {'Rank':<6} {'Score':>10} {'L-RMSD (Å)':>12} {'I-RMSD (Å)':>12}")
    print(f"  {'-------':<12} {'----':<6} {'-----':>10} {'----------':>12} {'----------':>12}")
    for b in all_benchmarks:
        lr = f"{b.lrmsd:.2f}" if b.lrmsd is not None else "  N/A"
        ir = f"{b.irmsd:.2f}" if b.irmsd is not None else "  N/A"
        print(f"  {b.complex_id:<12} {b.rank:<6} {b.score:>10.2f} {lr:>12} {ir:>12}")

    elapsed = time.time() - pipeline_start
    _banner(f"Pipeline complete in {elapsed:.1f} s  |  "
            f"{len(cases)} complex(es) processed  |  "
            f"Output → {args.output}/")


# ══════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "run.py",
        description = "Protein-RNA FFT Docking Pipeline — interactive entry point",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  python run.py --complex 1ASY
  python run.py --complex 1ASY --viz
  python run.py --complex 1ASY --viz structure grid rotations
  python run.py --complex 1ASY --viz structure
  python run.py --complex 1ASY --viz grid rotations
  python run.py --complex 1ASY 1AV6 1B23
  python run.py --all
  python run.py --all --step 15.0 --top_n 10
  python run.py --complex 1ASY --json "D:\\data\\PRDBv3_info.json" --pdb_root "D:\\data"
        """,
    )

    # ── Target selection: exactly one of --complex or --all is required ──
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--complex",
        nargs   = "+",
        metavar = "ID",
        help    = "One or more complex IDs to dock, e.g.  --complex 1ASY 1AV6 1B23",
    )
    target.add_argument(
        "--all",
        action  = "store_true",
        help    = "Run the pipeline on every UU case found in the JSON file",
    )

    # ── Visualization ────────────────────────────────────────────────────
    parser.add_argument(
        "--viz",
        nargs   = "*",        # 0 or more values:  --viz alone means "all",
        metavar = "TYPE",     # --viz structure grid  means just those two
        help    = (
            "Enable visualizations. Use alone (--viz) to show ALL types, "
            "or name specific ones: structure  grid  rotations. "
            "Examples:  --viz             → all three types; "
            "  --viz structure          → atom scatter only; "
            "  --viz grid rotations     → grids + SO(3) plots"
        ),
    )

    # ── Data paths ───────────────────────────────────────────────────────
    parser.add_argument(
        "--json",
        default = DEFAULT_JSON,
        help    = f"Path to PRDBv3_info.json  (default: {DEFAULT_JSON})",
    )
    parser.add_argument(
        "--pdb_root",
        default = DEFAULT_PDB_ROOT,
        help    = f"Root folder containing complex sub-folders  (default: {DEFAULT_PDB_ROOT})",
    )

    # ── Docking parameters ───────────────────────────────────────────────
    parser.add_argument(
        "--step",
        type    = float,
        default = 30.0,
        metavar = "DEG",
        help    = "Angular step size in degrees for SO(3) sampling  (default: 30.0)",
    )
    parser.add_argument(
        "--resolution",
        type    = float,
        default = 1.0,
        metavar = "Å",
        help    = "Voxel resolution in Ångström  (default: 1.0)",
    )
    parser.add_argument(
        "--top_n",
        type    = int,
        default = 5,
        metavar = "N",
        help    = "Number of top-ranked poses to export per complex  (default: 5)",
    )

    # ── Output paths ─────────────────────────────────────────────────────
    parser.add_argument(
        "--output",
        default = DEFAULT_OUTPUT,
        metavar = "DIR",
        help    = f"Root output directory for generated PDB files  (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--results",
        default = DEFAULT_RESULTS,
        metavar = "FILE",
        help    = f"Pickle file to save raw docking results  (default: {DEFAULT_RESULTS})",
    )

    return parser


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    # When --all is set, args.complex is None — normalise to empty list
    if args.all:
        args.complex = []

    # --viz with no arguments means "show all three types"
    if args.viz is not None and len(args.viz) == 0:
        args.viz = list(VIZ_CHOICES)

    # Validate any explicitly named viz types before doing any real work
    if args.viz:
        bad = [v for v in args.viz if v not in VIZ_CHOICES]
        if bad:
            print(f"[ERROR] Unknown --viz type(s): {bad}")
            print(f"        Valid choices are: {sorted(VIZ_CHOICES)}")
            sys.exit(1)

    _banner("Protein-RNA FFT Docking Pipeline", char="═")
    print(f"  Mode        : {'ALL cases' if not args.complex else str(args.complex)}")
    print(f"  JSON        : {args.json}")
    print(f"  PDB root    : {args.pdb_root}")
    print(f"  Angular step: {args.step}°   Resolution: {args.resolution} Å   Top-N: {args.top_n}")
    print(f"  Output dir  : {args.output}")
    if args.viz:
        print(f"  Visualize   : {sorted(args.viz)}")
    else:
        print(f"  Visualize   : off  (pass --viz to enable)")

    run_pipeline(args)
