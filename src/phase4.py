"""
Phase 4 — FFT Correlation Docking  (phase4.py)
===============================================
Executes the core FFT-based shape complementarity search.
All heavy computation runs on GPU:

    - RNA grid rotation  →  torch.nn.functional.grid_sample  (GPU)
    - FFT / IFFT         →  torch.fft.fftn / ifftn           (GPU)
    - Cross-correlation  →  pointwise complex multiply        (GPU)
    - Argmax             →  torch.argmax                      (GPU)

Sampling grids are computed in chunks (ROTATION_BATCH_SIZE) so the
approach works on any GPU from an RTX 3050 (8 GB) to an L40S (48 GB).

Tune the two constants below for your hardware.
"""

# ──────────────────────────────────────────────────────────────────────────
# Hardware constants — adjust for your GPU
# ──────────────────────────────────────────────────────────────────────────

# How many rotation sampling grids to build and hold in GPU memory at once.
# Each grid is  Nx × Ny × Nz × 3 × 4 bytes.
# On a 128³ grid one grid = 128³ × 3 × 4 ≈ 24 MB.
#
#   RTX 3050  8 GB  →  use  32
#   RTX 3090 24 GB  →  use 128
#   L40S     48 GB  →  use 256
#
ROTATION_BATCH_SIZE: int = 256

# Maximum grid dimension along any axis (powers of 2 only).
# Larger = more accurate but more VRAM and slower FFT.
#
#   RTX 3050  →  128
#   L40S      →  256
#
MAX_GRID_DIM: int = 256

# ──────────────────────────────────────────────────────────────────────────

import math
import time
import pickle
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from phase1 import load_cases, Structure
from phase2 import GridBuilder, _next_power_of_two, get_vdw_radius
from phase3 import SO3Sampler


@dataclass
class DockingResult:
    score:              float
    rotation_matrix:    np.ndarray
    translation_vector: np.ndarray   # Angstroms


# ═══════════════════════════════════════════════════════════════════════════
# Common Grid Manager  (unchanged from before)
# ═══════════════════════════════════════════════════════════════════════════

class CommonGridManager:
    """
    Embeds both protein and RNA into a shared grid large enough to
    prevent FFT circular convolution wrap-around artifacts.
    """

    def __init__(self, resolution: float = 1.0, padding: float = 10.0):
        self.resolution = resolution
        self.padding    = padding
        self.builder    = GridBuilder(resolution=resolution, padding=padding)

    def determine_common_shape(
        self,
        pro_coords: np.ndarray,
        rna_coords: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:

        p_min = pro_coords.min(axis=0)
        p_max = pro_coords.max(axis=0)

        rna_center   = rna_coords.mean(axis=0)
        rna_max_dist = np.max(np.linalg.norm(rna_coords - rna_center, axis=1))

        lo = p_min - rna_max_dist - self.padding
        hi = p_max + rna_max_dist + self.padding

        dims   = tuple(
            min(_next_power_of_two(math.ceil(s / self.resolution)), MAX_GRID_DIM)
            for s in (hi - lo)
        )
        origin = lo
        return origin, dims

    def build_protein_grid(
        self,
        struct: Structure,
        origin: np.ndarray,
        dims:   Tuple[int, int, int],
    ) -> np.ndarray:

        atoms  = self.builder._collect_atoms(struct, "protein")
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        radii  = np.array([get_vdw_radius(a) for a in atoms], dtype=np.float64)
        return self.builder._build_shape_grid(coords, radii, origin, dims)

    def build_rna_grid_native(
        self,
        struct: Structure,
        origin: np.ndarray,
        dims:   Tuple[int, int, int],
    ) -> np.ndarray:
        """Build the RNA shape grid ONCE at its native (unrotated) orientation."""
        atoms  = self.builder._collect_atoms(struct, "rna")
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        radii  = np.array([get_vdw_radius(a) for a in atoms], dtype=np.float64)
        return self.builder._build_shape_grid(coords, radii, origin, dims)


# ═══════════════════════════════════════════════════════════════════════════
# GPU grid rotation via grid_sample
# ═══════════════════════════════════════════════════════════════════════════

def _build_base_flat_grid(
    dims:   Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Build the normalised [-1, 1] coordinate grid for all voxels.
    Computed once and reused across every chunk.

    Returns flat_grid : (Nx*Ny*Nz, 3)  on device
    """
    Nx, Ny, Nz = dims
    xs = torch.linspace(-1, 1, Nz, device=device)   # W
    ys = torch.linspace(-1, 1, Ny, device=device)   # H
    zs = torch.linspace(-1, 1, Nx, device=device)   # D

    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")  # each (Nx,Ny,Nz)
    base_grid   = torch.stack([gx, gy, gz], dim=-1)          # (Nx,Ny,Nz,3)
    return base_grid.reshape(-1, 3)                           # (M, 3)


def build_sampling_grid_chunk(
    rotations:  List[np.ndarray],   # a slice of the full rotation list
    flat_grid:  torch.Tensor,        # (M, 3)  precomputed base grid
    dims:       Tuple[int, int, int],
    device:     torch.device,
) -> torch.Tensor:
    """
    Build sampling grids for a CHUNK of rotations only.

    Memory cost: chunk_size × Nx × Ny × Nz × 3 × 4 bytes
    At ROTATION_BATCH_SIZE=32, 128³ grid → 32 × 24 MB = 768 MB  (safe on 8 GB)

    Returns sampling_grids : (chunk_size, Nx, Ny, Nz, 3)  on device
    """
    Nx, Ny, Nz = dims
    B = len(rotations)

    R_inv = torch.tensor(
        np.stack([R.T for R in rotations], axis=0),   # (B, 3, 3)
        dtype=torch.float32,
        device=device,
    )

    # batched matmul:  (B, 3, 3) @ (B, 3, M)  →  (B, 3, M)  →  (B, M, 3)
    rotated_flat = torch.bmm(
        R_inv,
        flat_grid.unsqueeze(0).expand(B, -1, -1).permute(0, 2, 1),
    ).permute(0, 2, 1)                                # (B, M, 3)

    return rotated_flat.reshape(B, Nx, Ny, Nz, 3)


def rotate_grid_gpu(
    rna_tensor:      torch.Tensor,    # (1, 1, Nx, Ny, Nz)  on GPU
    sampling_grid:   torch.Tensor,    # (1, Nx, Ny, Nz, 3)  on GPU
) -> torch.Tensor:
    """
    Rotate the RNA grid using a precomputed sampling grid.
    Returns a rethresholded (Nx, Ny, Nz) tensor on GPU.
    """
    rotated = F.grid_sample(
        rna_tensor,
        sampling_grid,
        mode='bilinear',      # trilinear for 3D input
        padding_mode='zeros', # exterior = 0
        align_corners=True,
    )                         # → (1, 1, Nx, Ny, Nz)

    rotated = rotated.squeeze(0).squeeze(0)   # → (Nx, Ny, Nz)

    # Rethreshold: snap interpolated values back to {-15, 0, +1}
    out = torch.zeros_like(rotated)
    out[rotated >  0.5] =  1.0
    out[rotated < -0.5] = -15.0

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Core FFT Docking Engine
# ═══════════════════════════════════════════════════════════════════════════

class FFTDocker:

    def __init__(self, angular_step: float = 30.0, resolution: float = 1.0):
        self.resolution  = resolution
        self.sampler     = SO3Sampler(angular_step_deg=angular_step)
        self.grid_manager = CommonGridManager(resolution=resolution)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Hardware] FFTDocker using: {self.device.type.upper()}"
              + (f"  ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else ""))

    def dock(self, case) -> List[DockingResult]:
        print(f"\n[{case.complex_id}] Starting FFT Docking...")
        t0 = time.time()

        # ── Collect coordinates ──────────────────────────────────────────
        pro_atoms  = self.grid_manager.builder._collect_atoms(case.protein_struct, "protein")
        pro_coords = np.array([[a.x, a.y, a.z] for a in pro_atoms])

        rna_atoms  = self.grid_manager.builder._collect_atoms(case.rna_struct, "rna")
        rna_coords = np.array([[a.x, a.y, a.z] for a in rna_atoms])

        origin, dims = self.grid_manager.determine_common_shape(pro_coords, rna_coords)
        Nx, Ny, Nz   = dims
        print(f"  Common grid shape : {dims}  ({Nx*Ny*Nz:,} voxels)")

        # ── Build grids on CPU (once each) ───────────────────────────────
        print("  Building protein grid (CPU)...")
        pro_grid = self.grid_manager.build_protein_grid(case.protein_struct, origin, dims)

        print("  Building native RNA grid (CPU)...")
        rna_grid_native = self.grid_manager.build_rna_grid_native(case.rna_struct, origin, dims)

        # ── Push protein to GPU, precompute its FFT ──────────────────────
        pro_tensor = torch.tensor(pro_grid, dtype=torch.float32, device=self.device)
        fft_pro    = torch.fft.fftn(pro_tensor)

        # ── Push native RNA to GPU in grid_sample format ─────────────────
        rna_native_gpu = torch.tensor(
            rna_grid_native, dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(0)                         # (1,1,Nx,Ny,Nz)

        # ── Build base coordinate grid once (reused across all chunks) ───
        flat_grid = _build_base_flat_grid(dims, self.device)  # (M, 3)

        # ── Chunked rotation loop ─────────────────────────────────────────
        n_rots     = self.sampler.n_rotations
        all_rots   = self.sampler.rotations
        results    = []
        t_loop     = time.time()

        print(f"  Evaluating {n_rots:,} rotations "
              f"in chunks of {ROTATION_BATCH_SIZE}...")

        for chunk_start in range(0, n_rots, ROTATION_BATCH_SIZE):
            chunk_end  = min(chunk_start + ROTATION_BATCH_SIZE, n_rots)
            chunk_rots = all_rots[chunk_start:chunk_end]
            B          = len(chunk_rots)

            if chunk_start > 0 and chunk_start % (ROTATION_BATCH_SIZE * 10) == 0:
                elapsed = time.time() - t_loop
                eta     = elapsed / chunk_start * (n_rots - chunk_start)
                print(f"    {chunk_start:>6}/{n_rots}  "
                      f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

            # Build sampling grids for this chunk only — bounded VRAM
            sg_chunk = build_sampling_grid_chunk(
                chunk_rots, flat_grid, dims, self.device
            )                                               # (B, Nx, Ny, Nz, 3)

            for b in range(B):
                # ── Grid rotation (GPU) ──────────────────────────────────
                sg          = sg_chunk[b].unsqueeze(0)      # (1, Nx, Ny, Nz, 3)
                rna_rotated = rotate_grid_gpu(rna_native_gpu, sg)  # (Nx, Ny, Nz)

                # ── FFT cross-correlation (GPU) ──────────────────────────
                fft_rna = torch.fft.fftn(rna_rotated)
                corr    = torch.fft.ifftn(fft_pro * torch.conj(fft_rna)).real

                # ── Extract best translation (GPU) ───────────────────────
                best_flat  = torch.argmax(corr).item()
                best_score = corr.flatten()[best_flat].item()
                best_idx   = np.unravel_index(best_flat, dims)

                shift_x = best_idx[0] if best_idx[0] < Nx/2 else best_idx[0] - Nx
                shift_y = best_idx[1] if best_idx[1] < Ny/2 else best_idx[1] - Ny
                shift_z = best_idx[2] if best_idx[2] < Nz/2 else best_idx[2] - Nz

                translation = np.array([shift_x, shift_y, shift_z]) * self.resolution

                results.append(DockingResult(
                    score              = float(best_score),
                    rotation_matrix    = chunk_rots[b],
                    translation_vector = translation,
                ))

            # Free chunk grids immediately — keeps VRAM flat
            del sg_chunk

        results.sort(key=lambda x: x.score, reverse=True)

        elapsed = time.time() - t0
        print(f"\n[{case.complex_id}] Done in {elapsed:.1f}s  "
              f"({elapsed/n_rots*1000:.1f} ms/rotation)")
        print(f"  Top score: {results[0].score:.2f}")

        return results


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: FFT Docking")
    parser.add_argument("--json",       default="../assets/PRDBv3_updated.json")
    parser.add_argument("--pdb_root",   default="../ALL_PDBs")
    parser.add_argument("--step",       type=float, default=30.0)
    parser.add_argument("--resolution", type=float, default=1.0)
    args = parser.parse_args()

    cases, skipped = load_cases(args.json, args.pdb_root)
    if not cases:
        print("No cases loaded. Check paths.")
        exit()

    test_case = cases[0]
    docker    = FFTDocker(angular_step=args.step, resolution=args.resolution)
    results   = docker.dock(test_case)

    print("\nTop 5 Poses:")
    for i, r in enumerate(results[:5]):
        t = r.translation_vector
        print(f"  Rank {i+1}: score={r.score:>10.2f}  "
              f"t=[{t[0]:>6.1f}, {t[1]:>6.1f}, {t[2]:>6.1f}] Å")

    with open("../results.pkl", "wb") as f:
        pickle.dump({test_case.complex_id: results}, f)
    print("\nResults saved to ../results.pkl")