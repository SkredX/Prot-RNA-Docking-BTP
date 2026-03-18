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
    4.  Launch the selected module(s), each of which opens interactive Plotly
        browser tabs.

Modules
-------
    Module 1 (adv_channel_grids.py)  — Multi-channel grids
        Channel 0: Shape | Channel 1: Electrostatics | Channel 2: Desolvation

    Module 2 (adv_ion_grids.py)  — Explicit Ion Density Grids (Mg2+)
        Ion probability field | Charge screening | Volumetric ion-cloud

    Module 3 (adv_soft_grids.py)  — Soft Grids / Conformational Flexibility
        Gaussian blurring | 2D slices | Histograms | Penalty profiles

    Module 4 (adv_spf.py)  — Spherical Polar Fourier (SPF) Transforms
        SPF expansion | Power spectra | Overlap kernel | 3D sphere rendering

    Module 5 (adv_cnn_scoring.py)  — 3D-CNN AI Scoring Function
        Interface boxes | Architecture | Re-ranking | Grad-CAM saliency

Required files in this folder
------------------------------
    phase1.py, phase2.py, phase3.py   (copy from main pipeline)
    adv_channel_grids.py
    adv_ion_grids.py
    adv_soft_grids.py
    adv_spf.py
    adv_cnn_scoring.py
    adv_run.py

Dependencies
------------
    pip install numpy scipy plotly pydantic
    pip install torch          # Module 5 only
"""

import sys
import importlib
import builtins

sys.path.insert(0, ".")


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _prompt(msg: str, default: str = "") -> str:
    val = input(f"{msg} [{default}]: ").strip()
    return val if val else default


def _launch_module(mod_name: str, prefill_queue: list):
    """
    Import and run mod_name.main(), pre-filling the first N input() calls
    with the values in prefill_queue, then falling back to the real terminal.
    """
    mod = importlib.import_module(mod_name)
    _queue = list(prefill_queue)
    _real_input = builtins.input

    def _patched(prompt="", _q=_queue):
        if _q:
            val = _q.pop(0)
            print(f"{prompt}{val}")
            return val
        return _real_input(prompt)

    builtins.input = _patched
    try:
        mod.main()
    finally:
        builtins.input = _real_input


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    SEP  = "=" * 65
    THIN = "-" * 65

    print(f"\n{SEP}")
    print("  Protein-RNA Docking  |  Advanced Visualization Suite")
    print(f"{SEP}")
    print()
    print("  +-----------------------------------------------------+")
    print("  |  WARNING: This pipeline operates on ONE complex     |")
    print("  |  at a time. Each module opens interactive Plotly    |")
    print("  |  browser tabs -- close them when finished.          |")
    print("  +-----------------------------------------------------+")
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
    print(THIN)
    print("  [1]  Multi-Channel Grids")
    print("       Shape | Electrostatics | Desolvation shell")
    print()
    print("  [2]  Explicit Ion Density Grids  (Mg2+ / The Magnesium Problem)")
    print("       Ion probability field | Charge screening | Ion-cloud 3D")
    print()
    print("  [3]  Soft Grids -- Conformational Flexibility")
    print("       Gaussian blurring | 2D slices | Histograms | Penalty profiles")
    print()
    print("  [4]  Spherical Polar Fourier (SPF) Transforms")
    print("       Power spectra | Radial profiles | Overlap kernel | 3D sphere")
    print("       NOTE: ~15-30 s for SPF expansion step")
    print()
    print("  [5]  3D-CNN AI Scoring Function  (requires PyTorch)")
    print("       Interface boxes | CNN architecture | Re-ranking | Grad-CAM")
    print("       NOTE: Tensor extraction takes several minutes on CPU")
    print()
    print("  [A]  Run all five modules sequentially")
    print(f"{SEP}")

    choice = input("  Select module(s) (e.g.  1  or  1 3 4  or  A): ").strip().upper()

    module_map = {
        "1": "adv_channel_grids",
        "2": "adv_ion_grids",
        "3": "adv_soft_grids",
        "4": "adv_spf",
        "5": "adv_cnn_scoring",
    }

    if "A" in choice:
        modules_to_run = list(module_map.keys())
    else:
        modules_to_run = choice.split()

    valid = [m for m in modules_to_run if m in module_map]
    if not valid:
        print("  No valid module selected.  Exiting.")
        return

    for m in valid:
        mod_name = module_map[m]
        print(f"\n{SEP}")
        print(f"  Launching Module {m}: {mod_name}")
        print(f"{SEP}\n")

        # Module 4 needs an extra l_max prompt -- pre-fill with 8
        if m == "4":
            prefill = [complex_id, json_path, pdb_root, "8"]
        else:
            prefill = [complex_id, json_path, pdb_root]

        _launch_module(mod_name, prefill)

    print(f"\n{SEP}")
    print("  All selected modules completed.")
    print(f"  Complex processed: {complex_id}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
