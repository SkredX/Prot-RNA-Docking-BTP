"""
Advanced Visualization — Master Runner  (adv_run.py)
=====================================================
Interactive entry point for the entire Advanced Visualization pipeline.
Lets you pick which module(s) to run for a single complex.

Usage
-----
    python adv_run.py

The script will:
    1.  Print a welcome banner explaining this is a ONE-COMPLEX-AT-A-TIME tool.
    2.  Ask for the complex ID, JSON path, and PDB root (with sane defaults).
    3.  Present a module menu.
    4.  Launch the selected module, which opens Plotly visualisations in your
        default browser.

Modules
-------
    Module 1 (adv_channel_grids.py)  — Multi-channel grids
        • Channel 0: Shape (Katchalski-Katzir encoding)
        • Channel 1: Electrostatics (Poisson-Boltzmann / Debye-Hückel)
        • Channel 2: Desolvation shell (hydration-penalty layer ~3 Å)

    Module 2 (adv_ion_grids.py)  — Explicit Ion Density Grids (Mg²⁺)
        • Ion-probability voxel grid from negative-potential minima
        • Raw vs. Mg²⁺-screened electrostatics comparison
        • Volumetric ion-cloud rendering

    Module 3 (adv_soft_grids.py)  — Soft Grids / Conformational Flexibility
        • Gaussian-blurred potential (σ = 0.5, 1.0, 2.0 Å)
        • 2D cross-section slice comparisons (hard vs. soft)
        • Voxel-value histograms showing penalty redistribution
        • 3D gradient penalty rendering
        • Depth-vs-penalty profile curve

Required files in this folder
------------------------------
    phase1.py               ← copy from main pipeline
    phase2.py               ← copy from main pipeline
    adv_channel_grids.py
    adv_ion_grids.py
    adv_soft_grids.py
    adv_run.py              ← this file

Dependencies
------------
    pip install numpy scipy plotly pydantic
"""

import sys
import importlib

sys.path.insert(0, ".")


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _prompt(msg: str, default: str = "") -> str:
    val = input(f"{msg} [{default}]: ").strip()
    return val if val else default


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    SEP  = "═" * 65
    THIN = "─" * 65

    print(f"\n{SEP}")
    print("  Protein–RNA Docking  │  Advanced Visualization Suite")
    print(f"{SEP}")
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  ⚠  This pipeline operates on ONE complex at a time │")
    print("  │     Each module opens interactive Plotly browser     │")
    print("  │     tabs — close them when you are finished.         │")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    complex_id = input("  Enter complex ID  (e.g. 1ASY): ").strip().upper()
    if not complex_id:
        print("\n  No complex ID entered.  Exiting.")
        return

    print()
    json_path = _prompt(
        "  Path to PRDBv3_info.json",
        r"D:\BTP Files\PRDBv3.0\PRDBv3_info.json",
    )
    pdb_root = _prompt(
        "  PDB root folder      ",
        r"D:\BTP Files\PRDBv3.0",
    )

    print(f"\n{SEP}")
    print("  Module Menu")
    print(f"{THIN}")
    print("  [1]  Multi-Channel Grids")
    print("       → Shape | Electrostatics | Desolvation shell")
    print()
    print("  [2]  Explicit Ion Density Grids  (Mg²⁺ / The Magnesium Problem)")
    print("       → Ion probability field | Charge screening | Ion-cloud 3D")
    print()
    print("  [3]  Soft Grids — Conformational Flexibility")
    print("       → Gaussian blurring | 2D slices | Histograms | Penalty profiles")
    print()
    print("  [A]  Run all three modules sequentially")
    print(f"{SEP}")

    choice = input("  Select module(s) (e.g.  1  or  1 3  or  A): ").strip().upper()

    if "A" in choice:
        modules_to_run = ["1", "2", "3"]
    else:
        modules_to_run = choice.split()

    valid = [m for m in modules_to_run if m in {"1", "2", "3"}]
    if not valid:
        print("  No valid module selected.  Exiting.")
        return

    module_map = {
        "1": "adv_channel_grids",
        "2": "adv_ion_grids",
        "3": "adv_soft_grids",
    }

    # Inject the user-provided values so module mains() can use them
    # by monkey-patching sys.argv to simulate CLI arguments they expect
    # (all three modules use interactive prompts, not argparse, so we
    #  override stdin via a simple input-queue trick)

    for m in valid:
        mod_name = module_map[m]
        print(f"\n{'═'*65}")
        print(f"  Launching Module {m}: {mod_name}")
        print(f"{'═'*65}\n")

        mod = importlib.import_module(mod_name)

        # Patch input() to feed the complex_id, json_path, pdb_root
        # then fall through to the module's own interactive menu.
        _queue = [complex_id, json_path, pdb_root]

        original_input = __builtins__.__dict__["input"] if isinstance(__builtins__, dict) else input

        def _auto_input(prompt="", _q=_queue):
            if _q:
                val = _q.pop(0)
                print(f"{prompt}{val}")   # echo so user sees what was fed in
                return val
            return original_input(prompt)

        import builtins
        builtins.input = _auto_input

        try:
            mod.main()
        finally:
            builtins.input = original_input

    print(f"\n{SEP}")
    print("  All selected modules completed.")
    print(f"  Complex processed: {complex_id}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
