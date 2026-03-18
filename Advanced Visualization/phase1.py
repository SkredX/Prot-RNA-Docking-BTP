"""
Phase 1 — PDB Parser & Preprocessor  (phase1.py)
=================================================
Reads PRDBv3_info.json, filters UU docking cases, resolves PDB file paths
from the folder structure, and parses protein + RNA chains from each PDB.

Folder structure assumed based on local setup:
    D:\\BTP Files\\PRDBv3.0\\
        <C_PDB_ID>\\                  ← e.g., 1A9N
            <C_PDB_ID>.pdb            ← bound complex
            <U_pro_PDB_ID>.pdb        ← unbound protein
            <U_RNA_PDB_ID>.pdb        ← unbound RNA

Called by run.py — not normally executed directly.
"""

import os
import json
import argparse
from typing import Optional, List, Dict, Set
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

RNA_RESIDUES = {
    "A", "U", "G", "C",          
    "RA", "RU", "RG", "RC",      
    "ADE", "URA", "GUA", "CYT",  
}

PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL", "MSE",   
}


# ──────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────

class Atom(BaseModel):
    record: str       
    serial: int
    name: str         
    alt_loc: str
    res_name: str     
    chain_id: str
    res_seq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_factor: float = 0.0
    element: str = ""

class Chain(BaseModel):
    chain_id: str
    mol_type: str                 
    atoms: List[Atom] = Field(default_factory=list)

    def residue_names(self) -> Set[str]:
        return {a.res_name for a in self.atoms}

    def __repr__(self):
        return (
            f"Chain(id={self.chain_id!r}, type={self.mol_type!r}, "
            f"atoms={len(self.atoms)}, residues={len(self.residue_names())})"
        )

class Structure(BaseModel):
    pdb_id: str
    filepath: str
    chains: List[Chain] = Field(default_factory=list)

    def protein_chains(self):
        return [c for c in self.chains if c.mol_type == "protein"]

    def rna_chains(self):
        return [c for c in self.chains if c.mol_type == "rna"]

    def __repr__(self):
        return (
            f"Structure(pdb={self.pdb_id!r}, "
            f"chains={[c.chain_id for c in self.chains]}, "
            f"protein_chains={[c.chain_id for c in self.protein_chains()]}, "
            f"rna_chains={[c.chain_id for c in self.rna_chains()]})"
        )

class DockingCase(BaseModel):
    complex_id: str
    complex_pdb: str
    protein_pdb: str
    rna_pdb: str

    complex_struct: Optional[Structure] = None
    protein_struct: Optional[Structure] = None
    rna_struct: Optional[Structure] = None


# ──────────────────────────────────────────────
# Utilities & PDB Parser
# ──────────────────────────────────────────────

def clean_pdb_id(raw: str) -> str:
    return raw.strip().rstrip("*").strip()

def detect_mol_type(residue_names: Set[str]) -> str:
    n_protein = len(residue_names & PROTEIN_RESIDUES)
    n_rna = len(residue_names & RNA_RESIDUES)

    if n_protein == 0 and n_rna == 0:
        return "unknown"
    if n_protein >= n_rna:
        return "protein"
    return "rna"

def parse_pdb(filepath: str, pdb_id: str = "") -> Structure:
    chains_dict: Dict[str, List[Atom]] = {}

    with open(filepath, "r") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue

            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) > 76 else ""

            if atom_name.startswith("H") or element == "H":
                continue

            try:
                atom = Atom(
                    record=rec,
                    serial=int(line[6:11]),
                    name=atom_name,
                    alt_loc=line[16].strip(),
                    res_name=line[17:20].strip(),
                    chain_id=line[21].strip() or "_",
                    res_seq=int(line[22:26]),
                    icode=line[26].strip(),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    occupancy=float(line[54:60]) if line[54:60].strip() else 1.0,
                    b_factor=float(line[60:66]) if line[60:66].strip() else 0.0,
                    element=element,
                )
            except (ValueError, IndexError):
                continue

            if atom.alt_loc not in ("", "A", " "):
                continue

            chains_dict.setdefault(atom.chain_id, []).append(atom)

    chains: List[Chain] = []
    for chain_id, atoms in chains_dict.items():
        res_names = {a.res_name for a in atoms}
        mol_type = detect_mol_type(res_names)
        chains.append(Chain(chain_id=chain_id, mol_type=mol_type, atoms=atoms))

    return Structure(
        pdb_id=pdb_id or os.path.splitext(os.path.basename(filepath))[0],
        filepath=filepath,
        chains=chains,
    )


# ──────────────────────────────────────────────
# JSON loader & path resolver
# ──────────────────────────────────────────────

def resolve_pdb_path(pdb_root: str, complex_id: str, pdb_name: str) -> Optional[str]:
    # Ensure standard extension, assumes subfolders are named exactly as the complex ID
    filename = pdb_name.upper() + ".pdb"
    path = os.path.join(pdb_root, complex_id.upper(), filename)
    return path if os.path.isfile(path) else None

def load_uu_cases(json_path: str, pdb_root: str):
    # Ensure the JSON file exists before attempting to open it
    if not os.path.isfile(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return [], [{"reason": "JSON not found"}]

    with open(json_path, "r") as fh:
        records = json.load(fh)

    if isinstance(records, dict):
        for key in ("data", "entries", "complexes", "records"):
            if key in records:
                records = records[key]
                break
        else:
            records = next(v for v in records.values() if isinstance(v, list))

    cases: List[DockingCase] = []
    skipped = []

    for rec in records:
        docking_case = rec.get("Docking_case", "").strip().upper()

        if docking_case != "UU":
            continue

        complex_id = clean_pdb_id(rec.get("C_PDB", ""))
        pro_id = clean_pdb_id(rec.get("U_pro_PDB", ""))
        rna_id = clean_pdb_id(rec.get("U_RNA_PDB", ""))

        if not complex_id or not pro_id or not rna_id:
            skipped.append({"record": rec, "reason": "missing field"})
            continue

        complex_path = resolve_pdb_path(pdb_root, complex_id, complex_id)
        protein_path = resolve_pdb_path(pdb_root, complex_id, pro_id)
        rna_path = resolve_pdb_path(pdb_root, complex_id, rna_id)

        missing = []
        if not complex_path: missing.append(f"complex({complex_id})")
        if not protein_path: missing.append(f"protein({pro_id})")
        if not rna_path: missing.append(f"rna({rna_id})")

        if missing:
            skipped.append({"complex": complex_id, "reason": f"file not found: {missing}"})
            continue

        case = DockingCase(
            complex_id=complex_id,
            complex_pdb=complex_path,
            protein_pdb=protein_path,
            rna_pdb=rna_path,
        )

        try:
            case.complex_struct = parse_pdb(complex_path, pdb_id=complex_id)
            case.protein_struct = parse_pdb(protein_path, pdb_id=pro_id)
            case.rna_struct = parse_pdb(rna_path, pdb_id=rna_id)
        except Exception as e:
            skipped.append({"complex": complex_id, "reason": f"parse error: {e}"})
            continue

        cases.append(case)

    return cases, skipped


# ──────────────────────────────────────────────
# Validation & Output
# ──────────────────────────────────────────────

def validate_case(case: DockingCase):
    warnings = []
    if not case.protein_struct.protein_chains():
        warnings.append(f"[{case.complex_id}] Unbound protein file has NO detected protein chains")
    if not case.rna_struct.rna_chains():
        warnings.append(f"[{case.complex_id}] Unbound RNA file has NO detected RNA chains")

    pro_atoms = sum(len(c.atoms) for c in case.protein_struct.chains)
    rna_atoms = sum(len(c.atoms) for c in case.rna_struct.chains)

    if pro_atoms < 10: warnings.append(f"[{case.complex_id}] Very few protein atoms: {pro_atoms}")
    if rna_atoms < 10: warnings.append(f"[{case.complex_id}] Very few RNA atoms: {rna_atoms}")

    return warnings


def print_summary(cases, skipped):
    SEP = "─" * 70
    print(f"\n{'═'*70}")
    print("  Phase 1 Summary — UU Protein-RNA Docking Cases")
    print(f"{'═'*70}")
    print(f"  Loaded : {len(cases)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"{'═'*70}\n")

    for case in cases[:5]: # Truncated output to first 5 cases to avoid console spam locally
        print(SEP)
        print(f"  Complex  : {case.complex_id}")
        print("  Parsed structures")
        print(f"    Complex : {case.complex_struct}")
        print(f"    Protein : {case.protein_struct}")
        print(f"    RNA     : {case.rna_struct}")
        
        warnings = validate_case(case)
        if warnings:
            print("  ⚠ Warnings:")
            for w in warnings: print(f"    {w}")
        else:
            print("  ✓ Validation passed")
            
    if len(cases) > 5:
        print(f"\n  ... and {len(cases) - 5} more cases successfully loaded.")

    if skipped:
        print(f"\n{SEP}")
        print(f"  Skipped cases (Showing first 5 of {len(skipped)}):")
        for s in skipped[:5]:
            print(f"    {s}")
    print(f"\n{'═'*70}\n")


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def visualize_structure(struct: Structure, title: str = ""):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Run: pip install plotly")
        return

    protein_x, protein_y, protein_z = [], [], []
    rna_x, rna_y, rna_z = [], [], []
    unknown_x, unknown_y, unknown_z = [], [], []

    for chain in struct.chains:
        for atom in chain.atoms:
            if chain.mol_type == "protein":
                protein_x.append(atom.x)
                protein_y.append(atom.y)
                protein_z.append(atom.z)
            elif chain.mol_type == "rna":
                rna_x.append(atom.x)
                rna_y.append(atom.y)
                rna_z.append(atom.z)
            else:
                unknown_x.append(atom.x)
                unknown_y.append(atom.y)
                unknown_z.append(atom.z)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=protein_x, y=protein_y, z=protein_z, mode="markers", marker=dict(size=3), name="Protein"))
    fig.add_trace(go.Scatter3d(x=rna_x, y=rna_y, z=rna_z, mode="markers", marker=dict(size=3), name="RNA"))
    if unknown_x:
        fig.add_trace(go.Scatter3d(x=unknown_x, y=unknown_y, z=unknown_z, mode="markers", marker=dict(size=3), name="Unknown"))

    fig.update_layout(
        title=title or struct.pdb_id,
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
        width=900, height=700
    )
    fig.show()

# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Load and parse UU protein-RNA docking cases")

    # Updated defaults to use Windows raw strings pointing directly to your local BTP folder
    parser.add_argument(
        "--json",
        default=r"D:\BTP Files\PRDBv3.0\PRDBv3_info.json",
        help="Path to PRDBv3_info.json",
    )

    parser.add_argument(
        "--pdb_root",
        default=r"D:\BTP Files\PRDBv3.0",
        help="Root folder containing complex subfolders",
    )

    args = parser.parse_args()

    print(f"Loading JSON : {args.json}")
    print(f"PDB root     : {args.pdb_root}")

    cases, skipped = load_uu_cases(args.json, args.pdb_root)

    if cases:
        print("Opening visualization for first case in your default web browser...")
        visualize_structure(cases[0].complex_struct, title=f"Complex {cases[0].complex_id}")

    print_summary(cases, skipped)
    return cases

if __name__ == "__main__":
    main()
