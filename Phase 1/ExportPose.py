"""
ExportPose.py
Applies the winning rotation matrix and translation vector to the RNA
and exports a combined PDB file of the predicted complex.
"""

import numpy as np
import os
from PDBparser import Structure, Atom

def apply_transformation(rna_struct: Structure, R: np.ndarray, t: np.ndarray) -> Structure:
    # Find geometric center for rotation pivot
    atoms = [a for chain in rna_struct.chains for a in chain.atoms]
    coords = np.array([[a.x, a.y, a.z] for a in atoms])
    center = coords.mean(axis=0)

    # Apply Rotation and Translation to each atom
    for chain in rna_struct.chains:
        for atom in chain.atoms:
            orig_coord = np.array([atom.x, atom.y, atom.z])
            # R @ (coord - center) + center + translation
            new_coord = R @ (orig_coord - center) + center + t
            
            atom.x = float(new_coord[0])
            atom.y = float(new_coord[1])
            atom.z = float(new_coord[2])
            
    return rna_struct

def write_complex_pdb(protein_struct: Structure, transformed_rna: Structure, output_path: str):
    with open(output_path, 'w') as f:
        f.write(f"REMARK Predicted Complex\n")
        
        # Write Protein Atoms
        for chain in protein_struct.chains:
            for atom in chain.atoms:
                f.write(format_pdb_atom(atom))
                
        f.write("TER\n")
        
        # Write RNA Atoms
        for chain in transformed_rna.chains:
            for atom in chain.atoms:
                f.write(format_pdb_atom(atom))
                
        f.write("END\n")
    print(f"Successfully saved predicted complex to {output_path}")

def format_pdb_atom(a: Atom) -> str:
    # Strict 80-character PDB format column alignment
    return (f"{a.record:<6}{a.serial:>5} {a.name:<4}{a.alt_loc:>1}"
            f"{a.res_name:<3} {a.chain_id:>1}{a.res_seq:>4}{a.icode:>1}   "
            f"{a.x:>8.3f}{a.y:>8.3f}{a.z:>8.3f}"
            f"{a.occupancy:>6.2f}{a.b_factor:>6.2f}          {a.element:>2}\n")