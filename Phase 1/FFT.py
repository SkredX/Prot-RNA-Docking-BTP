"""
Phase 4 — FFT Correlation Docking (GPU Accelerated)
===================================
Executes the core FFT-based shape complementarity search using PyTorch 
to push grid arrays to the GPU, massively accelerating computation.
"""

import math
import time
import numpy as np
import argparse
import torch
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
        p_min = pro_coords.min(axis=0)
        p_max = pro_coords.max(axis=0)
        
        rna_center = rna_coords.mean(axis=0)
        rna_max_dist = np.max(np.linalg.norm(rna_coords - rna_center, axis=1))
        
        lo = p_min - rna_max_dist - self.padding
        hi = p_max + rna_max_dist + self.padding
        
        dims = tuple(_next_power_of_two(math.ceil(s / self.resolution)) for s in (hi - lo))
        origin = lo
        return origin, dims

    def build_protein_grid(self, struct: Structure, origin: np.ndarray, dims: Tuple[int, int, int]) -> np.ndarray:
        atoms = self.builder._collect_atoms(struct, "protein")
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        from grid3d import get_vdw_radius
        radii = np.array([get_vdw_radius(a) for a in atoms], dtype=np.float64)
        
        grid = self.builder._build_shape_grid(coords, radii, origin, dims)
        return grid

    def build_rna_grid(self, struct: Structure, origin: np.ndarray, dims: Tuple[int, int, int], current_coords: np.ndarray) -> np.ndarray:
        atoms = self.builder._collect_atoms(struct, "rna")
        from grid3d import get_vdw_radius
        radii = np.array([get_vdw_radius(a) for a in atoms], dtype=np.float64)
        
        grid = self.builder._build_shape_grid(current_coords, radii, origin, dims)
        ligand_grid = np.where(grid != 0, 1.0, 0.0).astype(np.float32)
        return ligand_grid


# ═══════════════════════════════════════════════════════════════════════════
# Core FFT Docking Engine (GPU Enabled)
# ═══════════════════════════════════════════════════════════════════════════

class FFTDocker:
    def __init__(self, angular_step: float = 30.0, resolution: float = 1.0):
        self.resolution = resolution
        self.sampler = SO3Sampler(angular_step_deg=angular_step)
        self.grid_manager = CommonGridManager(resolution=resolution)
        
        # 1. Initialize PyTorch Device
        # Automatically detects an Nvidia GPU. Falls back to CPU if none is found.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Hardware Initialization] FFTDocker using device: {self.device.type.upper()}")

    def dock(self, case) -> List[DockingResult]:
        print(f"\n[{case.complex_id}] Starting GPU-Accelerated FFT Docking...")
        start_time = time.time()

        pro_atoms = self.grid_manager.builder._collect_atoms(case.protein_struct, "protein")
        pro_coords = np.array([[a.x, a.y, a.z] for a in pro_atoms])
        
        rna_atoms = self.grid_manager.builder._collect_atoms(case.rna_struct, "rna")
        rna_coords = np.array([[a.x, a.y, a.z] for a in rna_atoms])
        rna_center = rna_coords.mean(axis=0)

        origin, dims = self.grid_manager.determine_common_shape(pro_coords, rna_coords)
        print(f"  Common Grid Shape : {dims}")

        print("  Building fixed protein grid...")
        pro_grid = self.grid_manager.build_protein_grid(case.protein_struct, origin, dims)
        
        # 2. Push Protein Grid to GPU and perform initial FFT
        pro_tensor = torch.tensor(pro_grid, dtype=torch.float32, device=self.device)
        fft_pro = torch.fft.fftn(pro_tensor)

        results = []
        n_rots = self.sampler.n_rotations

        print(f"  Evaluating {n_rots} rotations...")
        
        for i, R in enumerate(self.sampler):
            if i > 0 and i % 50 == 0:
                print(f"    Processed {i}/{n_rots} rotations...")

            # Rotate RNA around its own center (Runs on CPU - very fast)
            rot_rna_coords = rotate_coords(rna_coords, R, rna_center)

            # Build RNA grid (Runs on CPU)
            rna_grid = self.grid_manager.build_rna_grid(case.rna_struct, origin, dims, rot_rna_coords)
            
            # ──────────────────────────────────────────────────────────
            # GPU ACCELERATION BLOCK START
            # ──────────────────────────────────────────────────────────
            
            # Push dynamic RNA grid to GPU
            rna_tensor = torch.tensor(rna_grid, dtype=torch.float32, device=self.device)
            
            # Execute FFTs entirely in GPU memory
            fft_rna = torch.fft.fftn(rna_tensor)
            
            # Cross-correlation: IFFT( FFT(P) * conj(FFT(R)) )
            corr_tensor = torch.fft.ifftn(fft_pro * torch.conj(fft_rna)).real
            
            # Extract the max score and its flattened index directly from the GPU
            best_idx_flat = torch.argmax(corr_tensor).item()
            best_score = corr_tensor.flatten()[best_idx_flat].item()
            
            # Unravel the flat index back into 3D (x, y, z) using NumPy
            best_idx = np.unravel_index(best_idx_flat, dims)
            
            # ──────────────────────────────────────────────────────────
            # GPU ACCELERATION BLOCK END
            # ──────────────────────────────────────────────────────────

            shift_x = best_idx[0] if best_idx[0] < dims[0]/2 else best_idx[0] - dims[0]
            shift_y = best_idx[1] if best_idx[1] < dims[1]/2 else best_idx[1] - dims[1]
            shift_z = best_idx[2] if best_idx[2] < dims[2]/2 else best_idx[2] - dims[2]
            
            translation_vector = np.array([shift_x, shift_y, shift_z]) * self.resolution

            results.append(DockingResult(
                score=float(best_score),
                rotation_matrix=R,
                translation_vector=translation_vector
            ))

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
    
    parser.add_argument("--json",       default=r"D:\BTP Files\PRDBv3.0\PRDBv3_info.json")
    parser.add_argument("--pdb_root",   default=r"D:\BTP Files\PRDBv3.0")
    parser.add_argument("--step",       type=float, default=30.0, help="Angular step size (degrees)")
    parser.add_argument("--resolution", type=float, default=1.0, help="Grid voxel resolution")
    args = parser.parse_args()

    cases, skipped = load_uu_cases(args.json, args.pdb_root)
    if not cases:
        print("No cases loaded. Check paths.")
        exit()

    test_case = cases[0]
    docker = FFTDocker(angular_step=args.step, resolution=args.resolution)
    top_results = docker.dock(test_case)
    
    print("\nTop 5 Poses Found:")
    for idx, res in enumerate(top_results[:5]):
        print(f"  Pose {idx+1}: Score = {res.score:>8.2f} | Translation = [{res.translation_vector[0]:>6.1f}, {res.translation_vector[1]:>6.1f}, {res.translation_vector[2]:>6.1f}] Å")