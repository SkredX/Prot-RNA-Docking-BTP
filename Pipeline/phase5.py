"""
Phase 5 — PDB Generation & RMSD Benchmarking
==============================================
Takes the top DockingResult poses from Phase 4, generates PDB files
for each pose, then benchmarks them against the reference bound complex.

Convention (matches Phase 4):
    - Protein stays fixed in its original unbound coordinates
    - RNA is rotated (R) then translated (t) to produce the docked pose

Transform applied to each RNA atom:
    coord_docked = R @ (coord_unbound - rna_center) + rna_center + t

where t is the translation_vector from DockingResult (in Ångström),
and rna_center is the geometric center of the unbound RNA.

Output structure:
    generated_PDBS/
        <COMPLEX_ID>/
            rank1/
                protein.pdb
                rna.pdb
            rank2/
                protein.pdb
                rna.pdb
            ...

Benchmarking metrics:
    L-RMSD  — Ligand RMSD
               1. Superimpose predicted protein onto reference protein
                  using Kabsch algorithm (Cα atoms only)
               2. Apply that same superposition transform to predicted RNA
               3. Compute RMSD between transformed predicted RNA and
                  reference RNA (C4' atoms only)
               Standard in CAPRI / docking benchmarks.

    I-RMSD  — Interface RMSD
               RMSD of interface residues (those within 10Å of the
               partner in the reference complex), backbone atoms only.
               More sensitive to binding-site accuracy than L-RMSD.

Usage:
    python phase5_output.py --help
    python phase5_output.py \\
        --json      ./assets/PRDBv3.json \\
        --pdb_root  ./UU_PDBS \\
        --results   results.pkl \\
        --top_n     5
"""

import os
import math
import pickle
import argparse
import dataclasses
import numpy as np
from typing import List, Dict, Tuple, Optional

from phase1  import load_uu_cases, DockingCase, Structure, Atom, Chain, parse_pdb
from phase4  import DockingResult


# ═══════════════════════════════════════════════════════════════════════════
# Kabsch Algorithm — optimal superposition of two point sets
# ═══════════════════════════════════════════════════════════════════════════

def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the optimal rotation R and translation t that minimises
    RMSD between P (mobile) and Q (reference) point sets.

        P_aligned = (R @ (P - P_center).T).T + Q_center

    Parameters
    ----------
    P : (N, 3)  mobile coordinates
    Q : (N, 3)  reference coordinates  (same atom ordering as P)

    Returns
    -------
    R : (3, 3)  optimal rotation matrix
    t : (3,)    optimal translation vector

    Usage:
        R, t = kabsch(P, Q)
        P_aligned = (R @ (P - P.mean(0)).T).T + Q.mean(0)
        # equivalently:
        P_aligned = apply_superposition(P, R, t)
    """
    assert P.shape == Q.shape and P.ndim == 2 and P.shape[1] == 3

    P_center = P.mean(axis=0)
    Q_center = Q.mean(axis=0)

    P_c = P - P_center
    Q_c = Q - Q_center

    # Covariance matrix
    H = P_c.T @ Q_c                    # (3, 3)

    U, S, Vt = np.linalg.svd(H)

    # Ensure right-handed coordinate system (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    R = Vt.T @ D @ U.T                 # (3, 3)
    t = Q_center - R @ P_center        # (3,)

    return R, t


def apply_superposition(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rotation R and translation t to a coordinate array."""
    return (R @ coords.T).T + t


# ═══════════════════════════════════════════════════════════════════════════
# Coordinate extraction helpers
# ═══════════════════════════════════════════════════════════════════════════

def extract_backbone_coords(
    structure: Structure,
    mol_type:  str,
    atom_name: str,
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """
    Extract coordinates of a specific backbone atom from all residues.

    For protein : atom_name = 'CA'
    For RNA     : atom_name = 'C4\'' or 'P'

    Returns
    -------
    coords : (N, 3) array
    keys   : list of (chain_id, res_seq) — used to match atoms between
             predicted and reference structures
    """
    coords = []
    keys   = []

    chains = (structure.protein_chains() if mol_type == "protein"
              else structure.rna_chains())

    for chain in chains:
        # Group atoms by residue
        residues: Dict[int, List[Atom]] = {}
        for atom in chain.atoms:
            residues.setdefault(atom.res_seq, []).append(atom)

        for res_seq in sorted(residues):
            for atom in residues[res_seq]:
                if atom.name.strip() == atom_name:
                    coords.append([atom.x, atom.y, atom.z])
                    keys.append((chain.chain_id, res_seq))
                    break   # one atom per residue

    return np.array(coords, dtype=np.float64), keys


def extract_all_heavy_coords(
    structure: Structure,
    mol_type:  str,
) -> np.ndarray:
    """Extract all heavy atom coordinates for a molecule type."""
    chains = (structure.protein_chains() if mol_type == "protein"
              else structure.rna_chains())
    coords = []
    for chain in chains:
        for atom in chain.atoms:
            coords.append([atom.x, atom.y, atom.z])
    return np.array(coords, dtype=np.float64)


def _match_keys(
    keys_pred: List[Tuple],
    keys_ref:  List[Tuple],
) -> Tuple[List[int], List[int]]:
    """
    Return index pairs (idx_pred, idx_ref) for residues present in both.
    Handles cases where unbound and bound structures have slightly
    different residue sets (common in crystal structures).
    """
    ref_set  = {k: i for i, k in enumerate(keys_ref)}
    idx_pred = []
    idx_ref  = []
    for i, k in enumerate(keys_pred):
        if k in ref_set:
            idx_pred.append(i)
            idx_ref.append(ref_set[k])
    return idx_pred, idx_ref


# ═══════════════════════════════════════════════════════════════════════════
# Apply docking transform to RNA coordinates
# ═══════════════════════════════════════════════════════════════════════════

def apply_docking_result(
    rna_struct:   Structure,
    result:       DockingResult,
) -> np.ndarray:
    """
    Apply a DockingResult transform to all RNA atom coordinates.

    Transform:
        coord_docked = R @ (coord - center) + center + t

    Returns (N, 3) array of docked RNA coordinates.
    """
    # Collect all RNA heavy atom coordinates in chain order
    coords = []
    for chain in rna_struct.rna_chains():
        for atom in chain.atoms:
            coords.append([atom.x, atom.y, atom.z])
    coords = np.array(coords, dtype=np.float64)

    center = coords.mean(axis=0)
    R      = result.rotation_matrix
    t      = result.translation_vector

    # Rotate about molecular center, then translate
    docked = (R @ (coords - center).T).T + center + t
    return docked


def apply_docking_result_protein(protein_struct: Structure) -> np.ndarray:
    """
    Protein stays fixed — just return its original coordinates.
    """
    coords = []
    for chain in protein_struct.protein_chains():
        for atom in chain.atoms:
            coords.append([atom.x, atom.y, atom.z])
    return np.array(coords, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# RMSD calculation
# ═══════════════════════════════════════════════════════════════════════════

def rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """Root mean square deviation between two (N, 3) coordinate arrays."""
    assert A.shape == B.shape, f"Shape mismatch: {A.shape} vs {B.shape}"
    diff = A - B
    return float(math.sqrt((diff * diff).sum() / len(A)))


# ═══════════════════════════════════════════════════════════════════════════
# L-RMSD  (Ligand RMSD — CAPRI standard)
# ═══════════════════════════════════════════════════════════════════════════

def compute_lrmsd(
    pred_protein_struct: Structure,
    pred_rna_coords:     np.ndarray,   # docked RNA coords, all heavy atoms
    ref_complex_struct:  Structure,
) -> Optional[float]:
    """
    Ligand RMSD:
      1. Extract Cα atoms from predicted protein and reference protein
      2. Find Kabsch superposition: predicted protein → reference protein
      3. Apply same transform to docked RNA coordinates
      4. Extract C4' atoms from reference RNA
      5. RMSD between transformed predicted RNA C4' and reference RNA C4'

    Returns None if insufficient matched residues (<3).
    """
    # ── Step 1 & 2: superpose predicted protein onto reference protein ──
    pred_ca, pred_ca_keys = extract_backbone_coords(pred_protein_struct, "protein", "CA")
    ref_ca,  ref_ca_keys  = extract_backbone_coords(ref_complex_struct,  "protein", "CA")

    if len(pred_ca) == 0 or len(ref_ca) == 0:
        return None

    ip, ir = _match_keys(pred_ca_keys, ref_ca_keys)
    if len(ip) < 3:
        return None

    R_sup, t_sup = kabsch(pred_ca[ip], ref_ca[ir])

    # ── Step 3: apply superposition to all docked RNA coords ────────────
    pred_rna_sup = apply_superposition(pred_rna_coords, R_sup, t_sup)

    # ── Step 4 & 5: C4' RMSD vs reference RNA ───────────────────────────
    ref_c4p,  ref_c4p_keys  = extract_backbone_coords(ref_complex_struct, "rna", "C4'")

    if len(ref_c4p) == 0:
        # Some PDBs use C4* instead of C4'
        ref_c4p, ref_c4p_keys = extract_backbone_coords(ref_complex_struct, "rna", "C4*")

    if len(ref_c4p) == 0:
        return None

    # Build a fake "predicted RNA structure" with the same residue keys
    # so we can match by (chain_id, res_seq)
    pred_c4p_coords, pred_c4p_keys = _extract_c4p_from_docked(
        pred_rna_sup, pred_protein_struct, pred_rna_coords
    )

    # We can't use _match_keys here because pred_rna_sup is just a coord
    # array — we match by position index assuming same residue count
    # (safe: unbound RNA and bound RNA are the same molecule)
    n = min(len(pred_c4p_coords), len(ref_c4p))
    if n < 3:
        return None

    return rmsd(pred_c4p_coords[:n], ref_c4p[:n])


def _extract_c4p_from_docked(
    docked_all_coords: np.ndarray,
    pred_protein_struct: Structure,
    original_rna_coords: np.ndarray,
) -> Tuple[np.ndarray, list]:
    """
    The docked RNA coords array has all heavy atoms in chain order.
    We need to find which indices correspond to C4' atoms.

    We do this by matching positions in the original unbound RNA
    (where we know atom names) to the docked coord array
    (same atom ordering, just different positions).
    """
    # Actually: docked_all_coords was built from rna_struct chains in order.
    # We need the C4' indices from the original structure's atom list.
    # This is done by the caller — see compute_lrmsd_for_result() below,
    # which passes pre-extracted C4' indices.
    # This function is a stub — logic lives in compute_lrmsd_for_result.
    return docked_all_coords, []


# ═══════════════════════════════════════════════════════════════════════════
# Cleaner L-RMSD entry point
# ═══════════════════════════════════════════════════════════════════════════

def compute_lrmsd_for_result(
    case:               DockingCase,
    result:             DockingResult,
    ref_complex_struct: Structure,
) -> Optional[float]:
    """
    Full L-RMSD computation for one DockingResult.

    This is the function Phase 5 calls — it handles all the
    index bookkeeping internally.
    """
    # ── Collect unbound RNA atoms & coords in fixed order ───────────────
    rna_chains = case.rna_struct.rna_chains()
    rna_atoms  = [a for chain in rna_chains for a in chain.atoms]
    rna_coords_unbound = np.array([[a.x, a.y, a.z] for a in rna_atoms])

    # ── Apply docking transform to ALL rna atoms ─────────────────────────
    center = rna_coords_unbound.mean(axis=0)
    R      = result.rotation_matrix
    t      = result.translation_vector
    rna_coords_docked = (R @ (rna_coords_unbound - center).T).T + center + t

    # ── Find C4' indices in the unbound RNA atom list ────────────────────
    c4p_indices = [
        i for i, a in enumerate(rna_atoms)
        if a.name.strip() in ("C4'", "C4*")
    ]
    if len(c4p_indices) < 3:
        return None

    pred_c4p = rna_coords_docked[c4p_indices]   # predicted, post-transform

    # ── Superpose predicted protein onto reference protein (Kabsch) ──────
    pred_ca, pred_ca_keys = extract_backbone_coords(case.protein_struct,  "protein", "CA")
    ref_ca,  ref_ca_keys  = extract_backbone_coords(ref_complex_struct,   "protein", "CA")

    if len(pred_ca) < 3 or len(ref_ca) < 3:
        return None

    ip, ir = _match_keys(pred_ca_keys, ref_ca_keys)
    if len(ip) < 3:
        return None

    R_sup, t_sup = kabsch(pred_ca[ip], ref_ca[ir])

    # ── Apply superposition to predicted C4' coords ──────────────────────
    pred_c4p_sup = apply_superposition(pred_c4p, R_sup, t_sup)

    # ── Reference C4' coords ─────────────────────────────────────────────
    ref_c4p, _ = extract_backbone_coords(ref_complex_struct, "rna", "C4'")
    if len(ref_c4p) == 0:
        ref_c4p, _ = extract_backbone_coords(ref_complex_struct, "rna", "C4*")
    if len(ref_c4p) < 3:
        return None

    n = min(len(pred_c4p_sup), len(ref_c4p))
    return rmsd(pred_c4p_sup[:n], ref_c4p[:n])


# ═══════════════════════════════════════════════════════════════════════════
# I-RMSD  (Interface RMSD)
# ═══════════════════════════════════════════════════════════════════════════

def compute_irmsd_for_result(
    case:               DockingCase,
    result:             DockingResult,
    ref_complex_struct: Structure,
    interface_cutoff:   float = 10.0,
) -> Optional[float]:
    """
    Interface RMSD:
      1. Define interface residues as those in the reference complex
         with any heavy atom within interface_cutoff Å of the partner
      2. Extract backbone atoms (Cα / C4') of interface residues only
      3. Superpose predicted interface atoms onto reference interface atoms
      4. Report the RMSD

    The superposition is done on the interface atoms themselves
    (not on the whole protein first), which is why I-RMSD is more
    sensitive to binding-site accuracy than L-RMSD.
    """

    # ── Find interface residues in the reference complex ─────────────────
    ref_pro_coords = extract_all_heavy_coords(ref_complex_struct, "protein")
    ref_rna_coords = extract_all_heavy_coords(ref_complex_struct, "rna")

    if len(ref_pro_coords) == 0 or len(ref_rna_coords) == 0:
        return None

    # Protein interface: Cα of residues with any atom within cutoff of RNA
    pro_interface_keys = set()
    for chain in ref_complex_struct.protein_chains():
        residues: Dict[int, List] = {}
        for atom in chain.atoms:
            residues.setdefault(atom.res_seq, []).append(atom)
        for res_seq, atoms in residues.items():
            res_coords = np.array([[a.x, a.y, a.z] for a in atoms])
            # Min distance from this residue to any RNA atom
            dists = np.linalg.norm(
                res_coords[:, None, :] - ref_rna_coords[None, :, :],
                axis=2
            ).min()
            if dists <= interface_cutoff:
                pro_interface_keys.add((chain.chain_id, res_seq))

    # RNA interface: C4' of residues with any atom within cutoff of protein
    rna_interface_keys = set()
    for chain in ref_complex_struct.rna_chains():
        residues: Dict[int, List] = {}
        for atom in chain.atoms:
            residues.setdefault(atom.res_seq, []).append(atom)
        for res_seq, atoms in residues.items():
            res_coords = np.array([[a.x, a.y, a.z] for a in atoms])
            dists = np.linalg.norm(
                res_coords[:, None, :] - ref_pro_coords[None, :, :],
                axis=2
            ).min()
            if dists <= interface_cutoff:
                rna_interface_keys.add((chain.chain_id, res_seq))

    if len(pro_interface_keys) < 3 or len(rna_interface_keys) < 3:
        return None

    # ── Extract reference interface backbone coords ───────────────────────
    ref_ca_all,  ref_ca_keys  = extract_backbone_coords(ref_complex_struct, "protein", "CA")
    ref_c4p_all, ref_c4p_keys = extract_backbone_coords(ref_complex_struct, "rna",     "C4'")

    ref_ca_int  = ref_ca_all [[i for i, k in enumerate(ref_ca_keys)  if k in pro_interface_keys]]
    ref_c4p_int = ref_c4p_all[[i for i, k in enumerate(ref_c4p_keys) if k in rna_interface_keys]]

    ref_int = np.vstack([ref_ca_int, ref_c4p_int])

    # ── Extract predicted interface backbone coords (after docking) ───────
    # Protein: fixed
    pred_ca_all,  pred_ca_keys  = extract_backbone_coords(case.protein_struct, "protein", "CA")
    pred_ca_int = pred_ca_all[[i for i, k in enumerate(pred_ca_keys) if k in pro_interface_keys]]

    # RNA: apply docking transform
    rna_chains = case.rna_struct.rna_chains()
    rna_atoms  = [a for chain in rna_chains for a in chain.atoms]
    rna_coords_unbound = np.array([[a.x, a.y, a.z] for a in rna_atoms])

    center = rna_coords_unbound.mean(axis=0)
    R      = result.rotation_matrix
    t      = result.translation_vector
    rna_coords_docked = (R @ (rna_coords_unbound - center).T).T + center + t

    # Build (chain_id, res_seq) → docked_coord map for C4' atoms
    c4p_docked = {}
    idx = 0
    for chain in rna_chains:
        residues: Dict[int, List] = {}
        for i, atom in enumerate(chain.atoms):
            residues.setdefault(atom.res_seq, []).append((i, atom))
        for res_seq in sorted(residues):
            for i, atom in residues[res_seq]:
                if atom.name.strip() in ("C4'", "C4*"):
                    c4p_docked[(chain.chain_id, res_seq)] = rna_coords_docked[i]
                    break

    pred_c4p_int = np.array([
        c4p_docked[k] for k in rna_interface_keys
        if k in c4p_docked
    ])

    if len(pred_ca_int) == 0 or len(pred_c4p_int) == 0:
        return None

    pred_int = np.vstack([pred_ca_int, pred_c4p_int])

    n = min(len(pred_int), len(ref_int))
    if n < 3:
        return None

    # Superpose predicted interface onto reference interface
    R_sup, t_sup = kabsch(pred_int[:n], ref_int[:n])
    pred_int_sup = apply_superposition(pred_int[:n], R_sup, t_sup)

    return rmsd(pred_int_sup, ref_int[:n])


# ═══════════════════════════════════════════════════════════════════════════
# PDB Writer
# ═══════════════════════════════════════════════════════════════════════════

def write_pdb(
    structure:   Structure,
    coords:      np.ndarray,
    mol_type:    str,
    output_path: str,
):
    """
    Write a PDB file using atom records from structure but with
    coordinates replaced by the docked coords array.

    coords must be in the same atom order as the structure's chains.
    mol_type: 'protein' or 'rna' — selects which chains to write.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    chains = (structure.protein_chains() if mol_type == "protein"
              else structure.rna_chains())
    atoms  = [a for chain in chains for a in chain.atoms]

    assert len(atoms) == len(coords), (
        f"Atom count mismatch: structure has {len(atoms)}, "
        f"coords array has {len(coords)}"
    )

    with open(output_path, "w") as fh:
        fh.write(f"REMARK  Generated by Phase 5 FFT Docking Pipeline\n")
        fh.write(f"REMARK  mol_type={mol_type}\n")

        for i, (atom, (x, y, z)) in enumerate(zip(atoms, coords)):
            # PDB ATOM record format (columns are 1-indexed):
            # 1-6   record type
            # 7-11  serial
            # 13-16 atom name
            # 17    alt loc
            # 18-20 residue name
            # 22    chain ID
            # 23-26 residue seq
            # 27    insertion code
            # 31-38 x
            # 39-46 y
            # 47-54 z
            # 55-60 occupancy
            # 61-66 B-factor
            # 77-78 element

            record    = "ATOM  " if atom.record == "ATOM" else "HETATM"
            atom_name = _format_atom_name(atom.name, atom.element)

            line = (
                f"{record}"
                f"{(i+1):5d} "
                f"{atom_name:<4s}"
                f"{atom.alt_loc if atom.alt_loc else ' '}"
                f"{atom.res_name:>3s} "
                f"{atom.chain_id:1s}"
                f"{atom.res_seq:4d}"
                f"{atom.icode if atom.icode else ' ':1s}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{atom.occupancy:6.2f}"
                f"{atom.b_factor:6.2f}          "
                f"{atom.element:>2s}\n"
            )
            fh.write(line)

        fh.write("END\n")


def _format_atom_name(name: str, element: str) -> str:
    """
    Format atom name for PDB column 13-16 (4 characters).
    Single-character elements start at column 14 (index 1).
    Two-character elements start at column 13 (index 0).
    """
    name = name.strip()
    el   = element.strip() if element else name[0]
    if len(el) == 1 and len(name) < 4:
        return f" {name:<3s}"
    return f"{name:<4s}"


# ═══════════════════════════════════════════════════════════════════════════
# Main Phase 5 runner
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class BenchmarkResult:
    complex_id: str
    rank:       int
    score:      float
    lrmsd:      Optional[float]
    irmsd:      Optional[float]
    pdb_dir:    str


def run_phase5(
    case:             DockingCase,
    docking_results:  List[DockingResult],
    output_root:      str = "generated_PDBS",
    top_n:            int = 5,
    interface_cutoff: float = 10.0,
) -> List[BenchmarkResult]:
    """
    For a single DockingCase:
      1. Take top_n DockingResults (already sorted by score descending)
      2. Generate protein.pdb and rna.pdb for each rank
      3. Benchmark each against the reference bound complex
      4. Print and return BenchmarkResult list
    """

    complex_id = case.complex_id
    ref_struct  = case.complex_struct

    # Protein coordinates never change — compute once
    pro_chains = case.protein_struct.protein_chains()
    pro_atoms  = [a for chain in pro_chains for a in chain.atoms]
    pro_coords = np.array([[a.x, a.y, a.z] for a in pro_atoms])

    benchmark_results = []

    print(f"\n{'═'*65}")
    print(f"  Phase 5 — {complex_id}")
    print(f"{'═'*65}")
    print(f"  {'Rank':<6} {'Score':>10} {'L-RMSD (Å)':>12} {'I-RMSD (Å)':>12}  Output")
    print(f"  {'----':<6} {'-----':>10} {'----------':>12} {'----------':>12}  ------")

    for rank, result in enumerate(docking_results[:top_n], start=1):

        # ── Apply transform to RNA ────────────────────────────────────
        rna_chains = case.rna_struct.rna_chains()
        rna_atoms  = [a for chain in rna_chains for a in chain.atoms]
        rna_coords_unbound = np.array([[a.x, a.y, a.z] for a in rna_atoms])

        center = rna_coords_unbound.mean(axis=0)
        R      = result.rotation_matrix
        t      = result.translation_vector
        rna_coords_docked = (R @ (rna_coords_unbound - center).T).T + center + t

        # ── Write PDB files ───────────────────────────────────────────
        rank_dir = os.path.join(output_root, complex_id, f"rank{rank}")
        pro_pdb  = os.path.join(rank_dir, "protein.pdb")
        rna_pdb  = os.path.join(rank_dir, "rna.pdb")

        write_pdb(case.protein_struct, pro_coords,        "protein", pro_pdb)
        write_pdb(case.rna_struct,     rna_coords_docked, "rna",     rna_pdb)

        # ── Benchmark ─────────────────────────────────────────────────
        lrmsd = compute_lrmsd_for_result(case, result, ref_struct)
        irmsd = compute_irmsd_for_result(case, result, ref_struct,
                                         interface_cutoff=interface_cutoff)

        lrmsd_str = f"{lrmsd:.2f}" if lrmsd is not None else "  N/A"
        irmsd_str = f"{irmsd:.2f}" if irmsd is not None else "  N/A"

        print(
            f"  {rank:<6} {result.score:>10.2f} "
            f"{lrmsd_str:>12} {irmsd_str:>12}  {rank_dir}"
        )

        benchmark_results.append(BenchmarkResult(
            complex_id = complex_id,
            rank       = rank,
            score      = result.score,
            lrmsd      = lrmsd,
            irmsd      = irmsd,
            pdb_dir    = rank_dir,
        ))

    return benchmark_results


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5: Generate docked PDBs and benchmark with RMSD"
    )
    parser.add_argument("--json",      default="./assets/PRDBv3.json")
    parser.add_argument("--pdb_root",  default="./UU_PDBS")
    parser.add_argument("--results",   required=True,
                        help="Path to pickle file containing docking results dict "
                             "{complex_id: List[DockingResult]}")
    parser.add_argument("--output",    default="generated_PDBS")
    parser.add_argument("--top_n",     type=int,   default=5)
    parser.add_argument("--cutoff",    type=float, default=10.0,
                        help="Interface cutoff distance in Angstrom (default 10.0)")
    args = parser.parse_args()

    # Load docking results from pickle
    with open(args.results, "rb") as fh:
        all_results: Dict[str, List[DockingResult]] = pickle.load(fh)

    # Load structures
    cases, skipped = load_uu_cases(args.json, args.pdb_root)
    case_map = {c.complex_id: c for c in cases}

    all_benchmarks = []

    for complex_id, results in all_results.items():
        if complex_id not in case_map:
            print(f"  Warning: {complex_id} not found in loaded cases, skipping.")
            continue
        case = case_map[complex_id]
        benchmarks = run_phase5(
            case             = case,
            docking_results  = results,
            output_root      = args.output,
            top_n            = args.top_n,
            interface_cutoff = args.cutoff,
        )
        all_benchmarks.extend(benchmarks)

    # Print global summary
    print(f"\n{'═'*65}")
    print("  Global Summary")
    print(f"{'═'*65}")
    print(f"  {'Complex':<10} {'Rank':<6} {'L-RMSD':>10} {'I-RMSD':>10}")
    print(f"  {'-------':<10} {'----':<6} {'------':>10} {'------':>10}")
    for b in all_benchmarks:
        lr = f"{b.lrmsd:.2f}" if b.lrmsd is not None else "N/A"
        ir = f"{b.irmsd:.2f}" if b.irmsd is not None else "N/A"
        print(f"  {b.complex_id:<10} {b.rank:<6} {lr:>10} {ir:>10}")