"""
Phase 2 — 3D Shape Grid Builder
=================================
Converts a parsed Structure (from PDBparser.py) into a 3D voxel shape grid
ready for FFT correlation.

Encoding (Katchalski-Katzir 1992):
    interior voxels  → -15   (steric clash penalty)
    surface voxels   → +1    (accessible surface layer)
    exterior voxels  →  0

Grid dimensions are always powers of 2 for FFT efficiency.
"""

import math
import dataclasses
import numpy as np
import argparse
from typing import Tuple, Optional

# Import the necessary models and loader from your updated PDBparser.py
# Make sure your Phase 1 file is named 'PDBparser.py' and is in the same directory.
from PDBparser import Atom, Chain, Structure, load_uu_cases


# ═══════════════════════════════════════════════════════════════════════════
# VDW Radii  (Angstrom) — element-based
# Source: Bondi (1964) + AMBER99 supplement
# ═══════════════════════════════════════════════════════════════════════════

VDW_RADII: dict = {
    "C":  1.70, "N":  1.55, "O":  1.52, "S":  1.80, "P":  1.80,
    "H":  1.20, "F":  1.47, "CL": 1.75, "BR": 1.85, "I":  1.98,
    "SE": 1.90, "MG": 0.72, "ZN": 0.74, "CA": 1.00, "FE": 0.65,
    "MN": 0.83, "K":  1.38, "NA": 1.02,
}
DEFAULT_RADIUS = 1.70

def get_vdw_radius(atom: Atom) -> float:
    """Determine the Van der Waals radius of an atom based on its element."""
    el = atom.element.upper().strip() if atom.element else ""
    if el in VDW_RADII:
        return VDW_RADII[el]
    # Fallback: guess from the atom name (e.g., "CA" -> "C")
    name_el = atom.name.lstrip("0123456789")[0].upper() if atom.name else "C"
    return VDW_RADII.get(name_el, DEFAULT_RADIUS)


# ═══════════════════════════════════════════════════════════════════════════
# MolGrid — output container
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class MolGrid:
    """Stores the resulting 3D numpy array and spatial metadata."""
    pdb_id:     str
    mol_type:   str
    shape_grid: np.ndarray    # (Nx, Ny, Nz)
    origin:     np.ndarray    # (3,) Real-space origin in Angstrom
    resolution: float
    center:     np.ndarray    # (3,) Geometric center

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        return self.shape_grid.shape

    def voxel_to_coord(self, ix: int, iy: int, iz: int) -> np.ndarray:
        return self.origin + np.array([ix, iy, iz]) * self.resolution

    def coord_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        idx = ((np.array([x, y, z]) - self.origin) / self.resolution).astype(int)
        return tuple(np.clip(idx, 0, np.array(self.grid_shape) - 1))

    def summary(self) -> str:
        Nx, Ny, Nz = self.grid_shape
        n_surf = int((self.shape_grid  > 0).sum())
        n_int  = int((self.shape_grid  < 0).sum())
        return (
            f"MolGrid  pdb={self.pdb_id!r}  type={self.mol_type}\n"
            f"  grid shape   : ({Nx}, {Ny}, {Nz})  —  {Nx*Ny*Nz:,} voxels total\n"
            f"  resolution   : {self.resolution} Å/voxel\n"
            f"  origin       : ({self.origin[0]:.2f}, {self.origin[1]:.2f}, {self.origin[2]:.2f}) Å\n"
            f"  surface vox  : {n_surf:,}   (+1)\n"
            f"  interior vox : {n_int:,}   (-15)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GridBuilder
# ═══════════════════════════════════════════════════════════════════════════

class GridBuilder:
    """
    Constructs the Voxel grid. 
    Padding ensures there is enough 'empty' space around the molecule to slide 
    it around during the FFT correlation (Phase 4) without falling off the edge.
    """
    def __init__(
        self,
        resolution:        float = 1.0,
        padding:           float = 8.0,
        surface_thickness: float = 1.4, # 1.4A corresponds to water probe radius
        interior_penalty:  float = -15.0,
    ):
        self.resolution        = resolution
        self.padding           = padding
        self.surface_thickness = surface_thickness
        self.interior_penalty  = interior_penalty

    def build(self, structure: Structure, mol_type: str = "auto") -> MolGrid:
        atoms = self._collect_atoms(structure, mol_type)
        if not atoms:
            raise ValueError(f"No atoms found in {structure.pdb_id!r} for mol_type={mol_type!r}")

        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        radii  = np.array([get_vdw_radius(a) for a in atoms],  dtype=np.float64)
        center = coords.mean(axis=0)

        origin, grid_dims = self._compute_grid_dims(coords, radii)
        shape_grid = self._build_shape_grid(coords, radii, origin, grid_dims)

        if mol_type == "auto":
            n_pro = sum(len(c.atoms) for c in structure.protein_chains())
            n_rna = sum(len(c.atoms) for c in structure.rna_chains())
            mol_type_label = "protein" if n_pro >= n_rna else "rna"
        else:
            mol_type_label = mol_type

        return MolGrid(
            pdb_id     = structure.pdb_id,
            mol_type   = mol_type_label,
            shape_grid = shape_grid,
            origin     = origin,
            resolution = self.resolution,
            center     = center,
        )

    def _collect_atoms(self, structure, mol_type):
        if mol_type == "protein":
            chains = structure.protein_chains()
        elif mol_type == "rna":
            chains = structure.rna_chains()
        elif mol_type in ("auto", "all"):
            chains = structure.chains
        else:
            raise ValueError(f"Unknown mol_type: {mol_type!r}")
        return [atom for chain in chains for atom in chain.atoms]

    def _compute_grid_dims(self, coords, radii):
        max_r = radii.max()
        lo    = coords.min(axis=0) - max_r - self.padding
        hi    = coords.max(axis=0) + max_r + self.padding
        # Force grid dimensions to be powers of 2 for faster FFT math later
        dims  = tuple(_next_power_of_two(math.ceil(s / self.resolution)) for s in (hi - lo))
        return lo, dims

    def _build_shape_grid(self, coords, radii, origin, grid_dims):
        from scipy.ndimage import binary_erosion

        Nx, Ny, Nz = grid_dims
        grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        r    = self.resolution

        # Pass 1 — interior voxels
        for (ax, ay, az), radius in zip(coords, radii):
            lo_v = np.floor(((np.array([ax,ay,az]) - origin) - radius) / r).astype(int)
            hi_v = np.ceil (((np.array([ax,ay,az]) - origin) + radius) / r).astype(int) + 1
            lo_v = np.clip(lo_v, 0, [Nx-1, Ny-1, Nz-1])
            hi_v = np.clip(hi_v, 0, [Nx,   Ny,   Nz  ])

            ix = np.arange(lo_v[0], hi_v[0])
            iy = np.arange(lo_v[1], hi_v[1])
            iz = np.arange(lo_v[2], hi_v[2])

            dx = (origin[0] + ix*r - ax)[:, None, None]
            dy = (origin[1] + iy*r - ay)[None, :, None]
            dz = (origin[2] + iz*r - az)[None, None, :]

            mask = (dx**2 + dy**2 + dz**2) <= radius**2
            grid[lo_v[0]:hi_v[0], lo_v[1]:hi_v[1], lo_v[2]:hi_v[2]][mask] = self.interior_penalty

        # Pass 2 — surface voxels (Subtract the deeply buried interior from the outer bounds)
        interior_mask = grid < 0
        k             = max(1, round(self.surface_thickness / r))
        eroded        = binary_erosion(interior_mask, structure=_sphere_kernel(k), border_value=0)
        grid[interior_mask & ~eroded] = 1.0
        return grid


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n: p <<= 1
    return p

def _sphere_kernel(radius_vox: int) -> np.ndarray:
    d, c = 2*radius_vox+1, radius_vox
    k = np.zeros((d, d, d), dtype=bool)
    for ix in range(d):
        for iy in range(d):
            for iz in range(d):
                if (ix-c)**2+(iy-c)**2+(iz-c)**2 <= radius_vox**2:
                    k[ix, iy, iz] = True
    return k

def build_grids_for_case(case, builder: Optional[GridBuilder] = None):
    """Convenience wrapper to build protein + RNA shape grids for a given case."""
    if builder is None:
        builder = GridBuilder()
    return (
        builder.build(case.protein_struct, mol_type="protein"),
        builder.build(case.rna_struct,     mol_type="rna"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Grid Visualization
# ═══════════════════════════════════════════════════════════════════════════

def visualize_grid(grid: MolGrid, max_points: int = 150000):
    """
    Interactive 3D visualization of a MolGrid using Plotly.
    Surface voxels (+1)  → red, Interior voxels (-15) → blue
    Downsamples points if grid is massive to prevent crashing the browser.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Run: pip install plotly")
        return

    shape = grid.shape_grid
    surf = np.argwhere(shape > 0)
    interior = np.argwhere(shape < 0)

    # Downsample if too large
    if len(surf) > max_points:
        surf = surf[np.random.choice(len(surf), max_points, replace=False)]
    if len(interior) > max_points:
        interior = interior[np.random.choice(len(interior), max_points, replace=False)]

    surf_coords = grid.origin + surf * grid.resolution
    int_coords  = grid.origin + interior * grid.resolution

    fig = go.Figure()

    if len(int_coords) > 0:
        fig.add_trace(go.Scatter3d(
            x=int_coords[:,0], y=int_coords[:,1], z=int_coords[:,2],
            mode="markers", marker=dict(size=2, color='blue'), name="Interior (-15)"
        ))

    if len(surf_coords) > 0:
        fig.add_trace(go.Scatter3d(
            x=surf_coords[:,0], y=surf_coords[:,1], z=surf_coords[:,2],
            mode="markers", marker=dict(size=3, color='red'), name="Surface (+1)"
        ))

    fig.update_layout(
        title=f"Voxel Grid — {grid.pdb_id} ({grid.mol_type})",
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
        width=900, height=750
    )
    fig.show()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Build shape grids")
    
    # Matching your exact Windows file paths
    parser.add_argument("--json",       default=r"D:\BTP Files\PRDBv3.0\PRDBv3_info.json")
    parser.add_argument("--pdb_root",   default=r"D:\BTP Files\PRDBv3.0")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--padding",    type=float, default=8.0)
    args = parser.parse_args()

    # Leverage the PDBparser loader function directly
    cases, skipped = load_uu_cases(args.json, args.pdb_root)
    print(f"{len(cases)} cases loaded from Phase 1, {len(skipped)} skipped.\n")
    
    builder = GridBuilder(resolution=args.resolution, padding=args.padding)
    
    # Truncating to the first 3 cases so your local testing is fast and manageable
    print("Generating grids for the first 3 cases for testing...")
    for case in cases[:3]:
        print(f"{'─'*60}\n  Building Grid for Complex: {case.complex_id}")
        try:
            pg, rg = build_grids_for_case(case, builder)
            print(pg.summary())
            print(rg.summary())
            
            # Uncomment the line below if you want the 3D plot to open in your browser
            # visualize_grid(pg) 
            
        except Exception as e:
            print(f"  ✗ Error generating grid: {e}")
        print()
