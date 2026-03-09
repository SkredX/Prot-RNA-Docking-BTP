"""
Phase 4 — FFT Correlation Docking
===================================
Executes the core FFT-based shape complementarity search.

Algorithm:
    1. Define a common bounding box large enough to hold the Protein
       and the rotating RNA without circular wrap-around artifacts.
    2. Voxelize the fixed Protein (Receptor).
    3. Loop over SO(3) rotations:
        a. Rotate the RNA.
        b. Voxelize the rotated RNA (Ligand).
        c. Compute cross-correlation via FFT:
           score = IFFT( FFT(Protein) * conj(FFT(RNA)) )
        d. Find the translation (ix, iy, iz) that maximizes the score.
    4. Save the top-scoring pose for each rotation.

Standard Complementarity:
    Protein Grid: Surface = +1, Interior = -15, Empty = 0
    RNA Grid    : Surface/Interior = +1, Empty = 0
    (This ensures surface-surface overlap is positive, while clashes 
    with the protein interior generate massive negative penalties).
"""

import math
import time
import numpy as np
import argparse
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Local imports from your pipeline
from PDBparser import load_uu_cases, Structure
from grid3d import GridBuilder, MolGrid, _next_power_of_two
from RotSampler import SO3Sampler, rotate_coords


@dataclass
class DockingResult:
    score: float
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray  # in Angstroms


# ═══════════════════════════════════════════════════════════════════════════
# Common Grid Manager
# ═══════════════════════════════════════════════════════════════════════════

class CommonGridManager:
    """
    Ensures that both the Protein and RNA are embedded into a shared, 
    large enough grid to prevent FFT circular convolution artifacts.
    """
    def __init__(self, resolution: float = 1.0, padding: float = 10.0):
        self.resolution = resolution
        self.padding = padding
        self.builder = GridBuilder(resolution=resolution, padding=padding)

    def determine_common_shape(self, pro_coords: np.ndarray, rna_coords: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Calculates a bounding box that can contain the protein AND the RNA 
        orbiting around it.
        """
        # Get absolute bounds of protein
        p_min = pro_coords.min(axis=0)
        p_max = pro_coords.max(axis=0)
        
        # Max radius the RNA could sweep out if it orbits the protein
        rna_center = rna_coords.mean(axis=0)
        rna_max_dist = np.max(np.linalg.norm(rna_coords - rna_center, axis=1))
        
        # The common grid must hold the protein plus enough room for RNA to translate around it
        lo = p_min - rna_max_dist - self.padding
        hi = p_max + rna_max_dist + self.padding
        
        dims = tuple(_next_power_of_two(math.ceil(s / self.resolution)) for s in (hi - lo))
        origin = lo
        return origin, dims

    def build_protein_grid(self, struct: Structure, origin: np.ndarray, dims: Tuple[int, int, int]) -> np.ndarray:
        """Builds the static protein grid in the common box."""
        # Override the builder's dynamic dims to force our common dims
        atoms = self.builder._collect_atoms(struct, "protein")
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        from grid3d import get_vdw_radius
        radii = np.array([get_vdw_radius(a) for a in atoms], dtype=np.float64)
        
        grid = self.builder._build_shape_grid(coords, radii, origin, dims)
        return grid

    def build_rna_grid(self, struct: Structure, origin: np.ndarray, dims: Tuple[int, int, int], current_coords: np.ndarray) -> np.ndarray:
        """Builds the RNA grid and flattens it so interior and surface are both +1."""
        atoms = self.builder._collect_atoms(struct, "rna")
        from grid3d import get_vdw_radius
        radii = np.array([get_vdw_radius(a) for a in atoms], dtype=np.float64)
        
        grid = self.builder._build_shape_grid(current_coords, radii, origin, dims)
        
        # Modify Ligand grid for cross-correlation: 
        # Anything that belongs to the RNA (surface or interior) becomes 1.0
        ligand_grid = np.where(grid != 0, 1.0, 0.0).astype(np.float32)
        return ligand_grid


# ═══════════════════════════════════════════════════════════════════════════
# Core FFT Docking Engine
# ═══════════════════════════════════════════════════════════════════════════

class FFTDocker:
    def __init__(self, angular_step: float = 30.0, resolution: float = 1.0):
        self.resolution = resolution
        self.sampler = SO3Sampler(angular_step_deg=angular_step)
        self.grid_manager = CommonGridManager(resolution=resolution)

    def dock(self, case) -> List[DockingResult]:
        print(f"\n[{case.complex_id}] Starting FFT Docking...")
        start_time = time.time()

        # 1. Extract raw coordinates
        pro_atoms = self.grid_manager.builder._collect_atoms(case.protein_struct, "protein")
        pro_coords = np.array([[a.x, a.y, a.z] for a in pro_atoms])
        
        rna_atoms = self.grid_manager.builder._collect_atoms(case.rna_struct, "rna")
        rna_coords = np.array([[a.x, a.y, a.z] for a in rna_atoms])
        rna_center = rna_coords.mean(axis=0)

        # 2. Determine common grid bounding box
        origin, dims = self.grid_manager.determine_common_shape(pro_coords, rna_coords)
        print(f"  Common Grid Shape : {dims}")
        print(f"  Grid Origin       : {origin}")

        # 3. Build and pre-transform the static Protein Grid
        print("  Building fixed protein grid...")
        pro_grid = self.grid_manager.build_protein_grid(case.protein_struct, origin, dims)
        fft_pro = np.fft.fftn(pro_grid)

        results = []
        n_rots = self.sampler.n_rotations

        print(f"  Evaluating {n_rots} rotations...")
        
        # 4. Loop over all rotations
        for i, R in enumerate(self.sampler):
            if i > 0 and i % 50 == 0:
                print(f"    Processed {i}/{n_rots} rotations...")

            # Rotate RNA around its own center
            rot_rna_coords = rotate_coords(rna_coords, R, rna_center)

            # Build RNA grid
            rna_grid = self.grid_manager.build_rna_grid(case.rna_struct, origin, dims, rot_rna_coords)
            
            # FFT of RNA
            fft_rna = np.fft.fftn(rna_grid)

            # Cross-correlation: IFFT( FFT(P) * conj(FFT(R)) )
            # We use standard cross-correlation equation here. 
            corr_grid = np.real(np.fft.ifftn(fft_pro * np.conjugate(fft_rna)))

            # Find the grid voxel with the highest complementarity score
            best_idx = np.unravel_index(np.argmax(corr_grid), corr_grid.shape)
            best_score = corr_grid[best_idx]

            # Convert voxel shift to physical translation vector in Angstroms
            # Note: Because of how FFT wrap-around works, we must shift indices > N/2 to negative
            shift_x = best_idx[0] if best_idx[0] < dims[0]/2 else best_idx[0] - dims[0]
            shift_y = best_idx[1] if best_idx[1] < dims[1]/2 else best_idx[1] - dims[1]
            shift_z = best_idx[2] if best_idx[2] < dims[2]/2 else best_idx[2] - dims[2]
            
            translation_vector = np.array([shift_x, shift_y, shift_z]) * self.resolution

            results.append(DockingResult(
                score=float(best_score),
                rotation_matrix=R,
                translation_vector=translation_vector
            ))

        # Sort results by best score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        elapsed = time.time() - start_time
        print(f"[{case.complex_id}] Docking complete in {elapsed:.2f} seconds.")
        print(f"  Top Score: {results[0].score:.2f}")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: FFT Correlation")
    
    # Matching your exact Windows file paths
    parser.add_argument("--json",       default=r"D:\BTP Files\PRDBv3.0\PRDBv3_info.json")
    parser.add_argument("--pdb_root",   default=r"D:\BTP Files\PRDBv3.0")
    parser.add_argument("--step",       type=float, default=30.0, help="Angular step size (degrees)")
    parser.add_argument("--resolution", type=float, default=1.0, help="Grid voxel resolution")
    args = parser.parse_args()

    # Load from Phase 1
    cases, skipped = load_uu_cases(args.json, args.pdb_root)
    if not cases:
        print("No cases loaded. Check paths.")
        exit()

    # We will test on just the first case to avoid a long wait locally
    test_case = cases[0]
    
    # Initialize and run
    docker = FFTDocker(angular_step=args.step, resolution=args.resolution)
    
    top_results = docker.dock(test_case)
    
    print("\nTop 5 Poses Found:")
    for idx, res in enumerate(top_results[:5]):
        print(f"  Pose {idx+1}: Score = {res.score:>8.2f} | Translation = [{res.translation_vector[0]:>6.1f}, {res.translation_vector[1]:>6.1f}, {res.translation_vector[2]:>6.1f}] Å")