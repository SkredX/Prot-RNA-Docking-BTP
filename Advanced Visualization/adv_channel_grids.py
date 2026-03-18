"""
Advanced Visualization Module 1 — Multi-Channel Grids  (adv_channel_grids.py)
==============================================================================
Extends the single-channel shape grid (phase2.py) to a full multi-channel
voxel representation:

    Channel 0 — Shape         {-15, 0, +1}   (Katchalski-Katzir, unchanged)
    Channel 1 — Electrostatics (Poisson-Boltzmann approximation)
    Channel 2 — Desolvation    (hydration-shell penalty layer, ~3.0 Å)

All channels share the same grid origin, resolution, and dimensions as the
parent phase2 MolGrid so they can be directly correlated during FFT docking.

Standalone usage
----------------
    python adv_channel_grids.py

The script will:
  1. Prompt you for ONE complex ID (e.g. 1ASY).
  2. Ask for the JSON path and PDB root.
  3. Build all three channels for the protein AND RNA.
  4. Open Plotly visualisations in your browser (one tab per channel per molecule).

Dependencies
------------
    pip install numpy scipy plotly pydantic
    # phase1.py  and  phase2.py  must be present in the same folder.
"""

import sys
import math
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── pipeline imports (copy phase1.py & phase2.py into this folder) ──────────
sys.path.insert(0, ".")
from phase1 import load_uu_cases, Structure, Atom
from phase2 import GridBuilder, MolGrid, get_vdw_radius, _next_power_of_two


# ════════════════════════════════════════════════════════════════════════════
# Partial-charge lookup  (AMBER99 simplified, heavy atoms only)
# Source: Cornell et al. 1995 + standard AMBER topology
# Keys are (residue_name, atom_name) → partial charge in units of e
# Falls back to element-based heuristics for unrecognised residues.
# ════════════════════════════════════════════════════════════════════════════

_PARTIAL_CHARGES: dict = {
    # ── Backbone (protein) ──────────────────────────────────────────────────
    ("*", "N"):   -0.4157,
    ("*", "CA"):   0.0337,
    ("*", "C"):    0.5973,
    ("*", "O"):   -0.5679,
    # ── RNA phosphate (strong polyanion) ────────────────────────────────────
    ("*", "P"):    1.1662,
    ("*", "O1P"): -0.7761,
    ("*", "O2P"): -0.7761,
    ("*", "O5'"): -0.4954,
    ("*", "O3'"): -0.5246,
    ("*", "C5'"): -0.0069,
    ("*", "C4'"):  0.1065,
    ("*", "O4'"): -0.3548,
    ("*", "C1'"):  0.0394,
    ("*", "C2'"):  0.0670,
    ("*", "O2'"): -0.6139,
    # ── Common polar side-chains ─────────────────────────────────────────────
    ("LYS", "NZ"): -0.3854,
    ("ARG", "NH1"): -0.8627,
    ("ARG", "NH2"): -0.8627,
    ("ASP", "OD1"): -0.8014,
    ("ASP", "OD2"): -0.8014,
    ("GLU", "OE1"): -0.8188,
    ("GLU", "OE2"): -0.8188,
}

_ELEMENT_CHARGE_DEFAULTS: dict = {
    "N": -0.40, "O": -0.50, "P":  1.00,
    "C":  0.05, "S": -0.10, "MG": 2.00,
}


def get_partial_charge(atom: Atom) -> float:
    """Return the partial charge for an atom (e). Falls back gracefully."""
    key_res = (atom.res_name, atom.name)
    key_any = ("*", atom.name)
    if key_res in _PARTIAL_CHARGES:
        return _PARTIAL_CHARGES[key_res]
    if key_any in _PARTIAL_CHARGES:
        return _PARTIAL_CHARGES[key_any]
    el = (atom.element or atom.name[:1]).upper()
    return _ELEMENT_CHARGE_DEFAULTS.get(el, 0.0)


# ════════════════════════════════════════════════════════════════════════════
# MultiChannelGrid — data container
# ════════════════════════════════════════════════════════════════════════════

class MultiChannelGrid:
    """
    Stores all three voxel channels for one molecule.

    Attributes
    ----------
    pdb_id      : str
    mol_type    : str   — 'protein' | 'rna'
    origin      : np.ndarray (3,)
    resolution  : float
    shape_grid  : np.ndarray (Nx, Ny, Nz)   — Channel 0: KK shape
    elec_grid   : np.ndarray (Nx, Ny, Nz)   — Channel 1: electrostatics
    desolv_grid : np.ndarray (Nx, Ny, Nz)   — Channel 2: desolvation
    """

    def __init__(
        self,
        pdb_id: str,
        mol_type: str,
        shape_grid: np.ndarray,
        elec_grid: np.ndarray,
        desolv_grid: np.ndarray,
        origin: np.ndarray,
        resolution: float,
    ):
        self.pdb_id      = pdb_id
        self.mol_type    = mol_type
        self.shape_grid  = shape_grid
        self.elec_grid   = elec_grid
        self.desolv_grid = desolv_grid
        self.origin      = origin
        self.resolution  = resolution

    @property
    def grid_shape(self):
        return self.shape_grid.shape

    def summary(self) -> str:
        Nx, Ny, Nz = self.grid_shape
        n_surf  = int((self.shape_grid > 0).sum())
        n_int   = int((self.shape_grid < 0).sum())
        e_min   = float(self.elec_grid.min())
        e_max   = float(self.elec_grid.max())
        d_vox   = int((self.desolv_grid > 0).sum())
        return (
            f"\n{'═'*60}\n"
            f"  MultiChannelGrid  pdb={self.pdb_id!r}  type={self.mol_type}\n"
            f"{'─'*60}\n"
            f"  Grid shape   : ({Nx}, {Ny}, {Nz})\n"
            f"  Resolution   : {self.resolution} Å/voxel\n"
            f"  ── Channel 0 : Shape ──\n"
            f"    Surface vox  : {n_surf:,}  (+1)\n"
            f"    Interior vox : {n_int:,}  (-15)\n"
            f"  ── Channel 1 : Electrostatics ──\n"
            f"    Potential range : [{e_min:.3f}, {e_max:.3f}] kT/e\n"
            f"  ── Channel 2 : Desolvation ──\n"
            f"    Penalty voxels  : {d_vox:,}\n"
            f"{'═'*60}"
        )


# ════════════════════════════════════════════════════════════════════════════
# MultiChannelBuilder
# ════════════════════════════════════════════════════════════════════════════

class MultiChannelBuilder:
    """
    Builds all three grid channels for a single molecule (protein OR RNA).

    Parameters
    ----------
    resolution        : float   — voxel edge length in Å (default 1.0)
    padding           : float   — extra space around the molecule in Å (default 8.0)
    surface_thickness : float   — water probe radius defining surface shell (default 1.4 Å)
    hydration_radius  : float   — extent of desolvation shell beyond vdW surface (default 3.0 Å)
    debye_length      : float   — electrostatic screening length in Å (default 8.0 Å, ~150 mM salt)
    interior_penalty  : float   — KK interior value (default -15.0)
    """

    def __init__(
        self,
        resolution:        float = 1.0,
        padding:           float = 8.0,
        surface_thickness: float = 1.4,
        hydration_radius:  float = 3.0,
        debye_length:      float = 8.0,
        interior_penalty:  float = -15.0,
    ):
        self.resolution        = resolution
        self.padding           = padding
        self.surface_thickness = surface_thickness
        self.hydration_radius  = hydration_radius
        self.debye_length      = debye_length
        self.interior_penalty  = interior_penalty

        # Delegate shape-grid construction to the original GridBuilder
        self._shape_builder = GridBuilder(
            resolution=resolution,
            padding=padding,
            surface_thickness=surface_thickness,
            interior_penalty=interior_penalty,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def build(self, structure: Structure, mol_type: str) -> MultiChannelGrid:
        """Build all channels and return a MultiChannelGrid."""

        base_grid: MolGrid = self._shape_builder.build(structure, mol_type=mol_type)
        atoms = self._collect_atoms(structure, mol_type)

        coords  = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        charges = np.array([get_partial_charge(a) for a in atoms], dtype=np.float64)
        radii   = np.array([get_vdw_radius(a) for a in atoms],     dtype=np.float64)

        origin = base_grid.origin
        dims   = base_grid.grid_shape

        elec_grid   = self._build_electrostatic_grid(coords, charges, origin, dims)
        desolv_grid = self._build_desolvation_grid(base_grid.shape_grid, radii)

        return MultiChannelGrid(
            pdb_id      = structure.pdb_id,
            mol_type    = mol_type,
            shape_grid  = base_grid.shape_grid,
            elec_grid   = elec_grid,
            desolv_grid = desolv_grid,
            origin      = origin,
            resolution  = self.resolution,
        )

    # ── Channel 1: Poisson-Boltzmann electrostatics ─────────────────────────

    def _build_electrostatic_grid(
        self,
        coords:  np.ndarray,
        charges: np.ndarray,
        origin:  np.ndarray,
        dims:    tuple,
    ) -> np.ndarray:
        """
        Approximate PB electrostatic potential using Debye-Hückel screening.

        For each atom a with partial charge q_a at position r_a, the potential
        contribution at grid point r is:

            φ_a(r) = q_a * exp(-|r - r_a| / λ) / (ε * |r - r_a|)

        where λ = Debye screening length and ε = 4 (implicit protein interior).

        We use a cutoff of 4λ to keep the computation tractable.
        """
        Nx, Ny, Nz = dims
        r          = self.resolution
        cutoff     = 4.0 * self.debye_length   # Å
        cutoff_vox = int(math.ceil(cutoff / r))
        eps        = 4.0                        # effective dielectric

        grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)

        for (ax, ay, az), q in zip(coords, charges):
            if abs(q) < 1e-6:
                continue

            # voxel index of atom centre
            ix0 = int(round((ax - origin[0]) / r))
            iy0 = int(round((ay - origin[1]) / r))
            iz0 = int(round((az - origin[2]) / r))

            lo = np.clip([ix0 - cutoff_vox, iy0 - cutoff_vox, iz0 - cutoff_vox],
                         0, [Nx-1, Ny-1, Nz-1])
            hi = np.clip([ix0 + cutoff_vox + 1, iy0 + cutoff_vox + 1, iz0 + cutoff_vox + 1],
                         0, [Nx, Ny, Nz])

            ix = np.arange(lo[0], hi[0])
            iy = np.arange(lo[1], hi[1])
            iz = np.arange(lo[2], hi[2])

            dx = (origin[0] + ix * r - ax)[:, None, None]
            dy = (origin[1] + iy * r - ay)[None, :, None]
            dz = (origin[2] + iz * r - az)[None, None, :]

            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            # Avoid division by zero at atom centre
            dist = np.where(dist < 0.5, 0.5, dist)

            phi = q * np.exp(-dist / self.debye_length) / (eps * dist)
            grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += phi.astype(np.float32)

        return grid

    # ── Channel 2: Desolvation shell ─────────────────────────────────────────

    def _build_desolvation_grid(
        self,
        shape_grid: np.ndarray,
        radii:      np.ndarray,
    ) -> np.ndarray:
        """
        Build the desolvation penalty layer.

        The desolvation grid marks voxels that lie in the hydration shell —
        the region OUTSIDE the vdW surface but within `hydration_radius` Å
        of the molecular surface.  When a hydrophobic patch on the protein
        displaces this shell during docking, an energetic penalty is incurred.

        Implementation: binary-dilate the surface mask by k_hydration voxels,
        then subtract the original occupied (interior + surface) mask.
        Result voxels are +1 (penalty present) or 0 (open solvent).
        """
        k = max(1, round(self.hydration_radius / self.resolution))

        occupied_mask = shape_grid != 0          # interior + surface

        # Build a simple spherical structuring element of radius k voxels
        d = 2 * k + 1
        c = k
        struct = np.zeros((d, d, d), dtype=bool)
        for ix in range(d):
            for iy in range(d):
                for iz in range(d):
                    if (ix-c)**2 + (iy-c)**2 + (iz-c)**2 <= k**2:
                        struct[ix, iy, iz] = True

        dilated   = binary_dilation(occupied_mask, structure=struct)
        shell     = dilated & ~occupied_mask      # hydration shell only

        desolv = np.zeros_like(shape_grid, dtype=np.float32)
        desolv[shell] = 1.0
        return desolv

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _collect_atoms(self, structure: Structure, mol_type: str):
        if mol_type == "protein":
            chains = structure.protein_chains()
        elif mol_type == "rna":
            chains = structure.rna_chains()
        else:
            chains = structure.chains
        return [a for chain in chains for a in chain.atoms]


# ════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ════════════════════════════════════════════════════════════════════════════

def _sample_voxels(grid: np.ndarray, mask_fn, origin: np.ndarray, res: float,
                   max_pts: int = 60_000) -> np.ndarray:
    """Return (N, 3) real-space coordinates for voxels satisfying mask_fn."""
    indices = np.argwhere(mask_fn(grid))
    if len(indices) > max_pts:
        indices = indices[np.random.default_rng(42).choice(len(indices), max_pts, replace=False)]
    return origin + indices * res


def visualize_shape_channel(mcg: MultiChannelGrid, max_pts: int = 60_000):
    """Channel 0 — shape grid (KK encoding).  Red = surface, blue = interior."""
    surf = _sample_voxels(mcg.shape_grid, lambda g: g > 0, mcg.origin, mcg.resolution, max_pts)
    intr = _sample_voxels(mcg.shape_grid, lambda g: g < 0, mcg.origin, mcg.resolution, max_pts)

    fig = go.Figure()
    if len(intr):
        fig.add_trace(go.Scatter3d(x=intr[:,0], y=intr[:,1], z=intr[:,2],
                                   mode="markers", marker=dict(size=2, color="royalblue", opacity=0.5),
                                   name="Interior (−15)"))
    if len(surf):
        fig.add_trace(go.Scatter3d(x=surf[:,0], y=surf[:,1], z=surf[:,2],
                                   mode="markers", marker=dict(size=3, color="tomato", opacity=0.8),
                                   name="Surface (+1)"))
    fig.update_layout(
        title=f"Channel 0 — Shape Grid  |  {mcg.pdb_id}  ({mcg.mol_type})",
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
        width=950, height=750,
    )
    fig.show()


def visualize_electrostatic_channel(mcg: MultiChannelGrid, max_pts: int = 60_000,
                                     percentile_clip: float = 2.0):
    """
    Channel 1 — electrostatic potential.
    Colour scale: blue (negative/anionic) → white (neutral) → red (positive/cationic).
    Only non-zero voxels within the Debye cutoff are shown.
    """
    nonzero = np.argwhere(np.abs(mcg.elec_grid) > 1e-4)
    if len(nonzero) == 0:
        print("  [Warning] Electrostatic grid is entirely zero — no atoms with charge?")
        return

    if len(nonzero) > max_pts:
        nonzero = nonzero[np.random.default_rng(42).choice(len(nonzero), max_pts, replace=False)]

    coords = mcg.origin + nonzero * mcg.resolution
    vals   = mcg.elec_grid[nonzero[:,0], nonzero[:,1], nonzero[:,2]]

    # Robust colour clipping
    vlo = np.percentile(vals, percentile_clip)
    vhi = np.percentile(vals, 100 - percentile_clip)

    fig = go.Figure(go.Scatter3d(
        x=coords[:,0], y=coords[:,1], z=coords[:,2],
        mode="markers",
        marker=dict(
            size=3,
            color=vals,
            colorscale="RdBu_r",
            cmin=vlo, cmax=vhi,
            colorbar=dict(title="φ (kT/e)"),
            opacity=0.7,
        ),
        name="Electrostatic potential",
    ))
    fig.update_layout(
        title=f"Channel 1 — Electrostatic Potential (PB approx.)  |  {mcg.pdb_id}  ({mcg.mol_type})",
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
        width=950, height=750,
    )
    fig.show()


def visualize_desolvation_channel(mcg: MultiChannelGrid, max_pts: int = 60_000):
    """Channel 2 — desolvation / hydration shell (orange voxels)."""
    shell = _sample_voxels(mcg.desolv_grid, lambda g: g > 0, mcg.origin, mcg.resolution, max_pts)
    surf  = _sample_voxels(mcg.shape_grid,  lambda g: g > 0, mcg.origin, mcg.resolution, max_pts // 2)

    fig = go.Figure()
    if len(surf):
        fig.add_trace(go.Scatter3d(x=surf[:,0], y=surf[:,1], z=surf[:,2],
                                   mode="markers", marker=dict(size=2, color="steelblue", opacity=0.3),
                                   name="Molecular surface"))
    if len(shell):
        fig.add_trace(go.Scatter3d(x=shell[:,0], y=shell[:,1], z=shell[:,2],
                                   mode="markers", marker=dict(size=2, color="darkorange", opacity=0.5),
                                   name="Hydration shell (penalty)"))
    fig.update_layout(
        title=f"Channel 2 — Desolvation Shell (~3 Å)  |  {mcg.pdb_id}  ({mcg.mol_type})",
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
        width=950, height=750,
    )
    fig.show()


def visualize_all_channels_overlay(mcg: MultiChannelGrid, max_pts: int = 30_000):
    """
    Single-figure overlay: shape surface (blue), desolvation shell (orange),
    electrostatic potential (RdBu colour-mapped, surface voxels only).
    Useful for seeing the spatial relationship of all three channels at once.
    """
    surf_idx  = np.argwhere(mcg.shape_grid > 0)
    shell_idx = np.argwhere(mcg.desolv_grid > 0)

    rng = np.random.default_rng(0)
    if len(surf_idx)  > max_pts: surf_idx  = surf_idx [rng.choice(len(surf_idx),  max_pts, replace=False)]
    if len(shell_idx) > max_pts: shell_idx = shell_idx[rng.choice(len(shell_idx), max_pts, replace=False)]

    surf_coords  = mcg.origin + surf_idx  * mcg.resolution
    shell_coords = mcg.origin + shell_idx * mcg.resolution

    surf_phi = mcg.elec_grid[surf_idx[:,0], surf_idx[:,1], surf_idx[:,2]]
    vlo = np.percentile(surf_phi, 2)
    vhi = np.percentile(surf_phi, 98)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=surf_coords[:,0], y=surf_coords[:,1], z=surf_coords[:,2],
        mode="markers",
        marker=dict(size=3, color=surf_phi, colorscale="RdBu_r",
                    cmin=vlo, cmax=vhi, colorbar=dict(title="φ (kT/e)"), opacity=0.8),
        name="Surface + Electrostatics",
    ))
    fig.add_trace(go.Scatter3d(
        x=shell_coords[:,0], y=shell_coords[:,1], z=shell_coords[:,2],
        mode="markers",
        marker=dict(size=2, color="darkorange", opacity=0.25),
        name="Hydration shell",
    ))
    fig.update_layout(
        title=f"All-Channel Overlay  |  {mcg.pdb_id}  ({mcg.mol_type})",
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
        width=1000, height=800,
    )
    fig.show()


# ════════════════════════════════════════════════════════════════════════════
# Interactive entry point
# ════════════════════════════════════════════════════════════════════════════

def _prompt(msg: str, default: str = "") -> str:
    val = input(f"{msg} [{default}]: ").strip()
    return val if val else default


def main():
    SEP = "═" * 65
    print(f"\n{SEP}")
    print("  Advanced Visualization — Module 1: Multi-Channel Grids")
    print(f"  Electrostatics  |  Desolvation Shell")
    print(f"{SEP}")
    print("  ⚠  This pipeline processes ONE complex at a time.")
    print(f"{SEP}\n")

    complex_id = input("  Enter complex ID (e.g. 1ASY): ").strip().upper()
    if not complex_id:
        print("No complex ID entered. Exiting.")
        return

    json_path = _prompt(
        "  Path to PRDBv3_info.json",
        r"D:\BTP Files\PRDBv3.0\PRDBv3_info.json",
    )
    pdb_root = _prompt(
        "  PDB root folder",
        r"D:\BTP Files\PRDBv3.0",
    )

    print(f"\n  Loading cases from JSON …")
    cases, skipped = load_uu_cases(json_path, pdb_root)

    # Filter to requested complex
    target = next((c for c in cases if c.complex_id == complex_id), None)
    if target is None:
        print(f"\n  ✗ Complex '{complex_id}' not found among {len(cases)} UU cases.")
        print(f"    Skipped: {len(skipped)}  |  check paths and ID.")
        return

    print(f"  ✓ Loaded complex {complex_id}\n")

    builder = MultiChannelBuilder(resolution=1.0, padding=8.0)

    print("  Building multi-channel grids …")
    print("    → Protein …")
    pro_mcg = builder.build(target.protein_struct, mol_type="protein")
    print(pro_mcg.summary())

    print("\n    → RNA …")
    rna_mcg = builder.build(target.rna_struct, mol_type="rna")
    print(rna_mcg.summary())

    print(f"\n{SEP}")
    print("  Visualization Menu")
    print("  ─────────────────")
    print("  [1]  Channel 0 — Shape grid (protein)")
    print("  [2]  Channel 0 — Shape grid (RNA)")
    print("  [3]  Channel 1 — Electrostatics (protein)")
    print("  [4]  Channel 1 — Electrostatics (RNA)")
    print("  [5]  Channel 2 — Desolvation shell (protein)")
    print("  [6]  Channel 2 — Desolvation shell (RNA)")
    print("  [7]  All-channel overlay (protein)")
    print("  [8]  All-channel overlay (RNA)")
    print("  [A]  All of the above (8 browser tabs)")
    print(f"{SEP}")

    choice = input("  Enter option(s) separated by spaces (e.g.  3 4  or  A): ").strip().upper()

    dispatch = {
        "1": lambda: visualize_shape_channel(pro_mcg),
        "2": lambda: visualize_shape_channel(rna_mcg),
        "3": lambda: visualize_electrostatic_channel(pro_mcg),
        "4": lambda: visualize_electrostatic_channel(rna_mcg),
        "5": lambda: visualize_desolvation_channel(pro_mcg),
        "6": lambda: visualize_desolvation_channel(rna_mcg),
        "7": lambda: visualize_all_channels_overlay(pro_mcg),
        "8": lambda: visualize_all_channels_overlay(rna_mcg),
    }

    if "A" in choice:
        keys = ["1", "2", "3", "4", "5", "6", "7", "8"]
    else:
        keys = choice.split()

    valid = [k for k in keys if k in dispatch]
    if not valid:
        print("  No valid options selected. Exiting.")
        return

    print(f"\n  Opening {len(valid)} visualisation(s) …")
    for k in valid:
        print(f"    → Option {k}")
        dispatch[k]()

    print(f"\n{SEP}")
    print("  Done.  Close browser tabs when finished.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
