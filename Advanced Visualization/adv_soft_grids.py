"""
Advanced Visualization Module 3 — Soft Grids for Conformational Flexibility  (adv_soft_grids.py)
==================================================================================================
Replaces the hard Katchalski-Katzir -15 clash penalty with a smooth,
Gaussian-blurred potential that linearly penalises minor steric overlaps.

Motivation
----------
Rigid-body docking fails when molecules undergo "induced-fit" conformational
changes (typically side-chain rotations of ±1–2 Å) upon binding.  The hard
-15 interior penalty rejects ANY overlap — even biologically acceptable ones.

Soft-potential docking (Gaussian blurring) addresses this by:
    • Penalising deep clashes (interior overlap > Δ Å) heavily (nearly -15)
    • Penalising shallow clashes (surface overlap ~1 Å) linearly (~−2)
    • Leaving genuine surface contacts unchanged (+1)

This module:
    1.  Builds the standard hard-potential grid (baseline).
    2.  Applies three Gaussian blur sigma values (σ = 0.5, 1.0, 2.0 Å) to
        demonstrate the softening continuum.
    3.  Visualises 2D cross-sectional slices through the grid (fast and
        informative), plus a 3D surface/interior comparison.
    4.  Plots the voxel-value histogram before and after softening to make
        the redistribution of penalty mass visible.

Standalone usage
----------------
    python adv_soft_grids.py

    The script will:
      1.  Prompt for ONE complex ID.
      2.  Ask for JSON and PDB root paths.
      3.  Build grids for protein AND RNA.
      4.  Open interactive Plotly visualisations in browser tabs.

Dependencies
------------
    pip install numpy scipy plotly pydantic
    # phase1.py  and  phase2.py  must be present in the same folder.
"""

import sys
import math
import numpy as np
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, ".")
from phase1 import load_uu_cases, Structure
from phase2 import GridBuilder, MolGrid


# ════════════════════════════════════════════════════════════════════════════
# SoftGrid — data container
# ════════════════════════════════════════════════════════════════════════════

class SoftGrid:
    """
    Stores the hard grid and three Gaussian-softened variants for comparison.

    Attributes
    ----------
    pdb_id     : str
    mol_type   : str
    origin     : np.ndarray (3,)
    resolution : float
    hard_grid  : np.ndarray  — original {-15, 0, +1} encoding
    soft_grids : dict        — {sigma_angstrom: np.ndarray}  softened grids
    """

    def __init__(
        self,
        pdb_id:     str,
        mol_type:   str,
        origin:     np.ndarray,
        resolution: float,
        hard_grid:  np.ndarray,
        soft_grids: dict,
    ):
        self.pdb_id     = pdb_id
        self.mol_type   = mol_type
        self.origin     = origin
        self.resolution = resolution
        self.hard_grid  = hard_grid
        self.soft_grids = soft_grids   # {0.5: arr, 1.0: arr, 2.0: arr}

    @property
    def grid_shape(self):
        return self.hard_grid.shape

    def summary(self) -> str:
        Nx, Ny, Nz = self.grid_shape
        n_surf = int((self.hard_grid > 0).sum())
        n_int  = int((self.hard_grid < 0).sum())
        lines  = [
            f"\n{'═'*60}",
            f"  SoftGrid  pdb={self.pdb_id!r}  type={self.mol_type}",
            f"{'─'*60}",
            f"  Grid shape   : ({Nx}, {Ny}, {Nz})",
            f"  Resolution   : {self.resolution} Å/voxel",
            f"  ── Hard grid ──",
            f"    Surface vox  : {n_surf:,}  (+1)",
            f"    Interior vox : {n_int:,}  (-15)",
            f"  ── Soft grids ──",
        ]
        for sigma, sg in self.soft_grids.items():
            lines.append(
                f"    σ={sigma:.1f} Å  →  value range [{sg.min():.2f}, {sg.max():.2f}]"
            )
        lines.append(f"{'═'*60}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# SoftGridBuilder
# ════════════════════════════════════════════════════════════════════════════

class SoftGridBuilder:
    """
    Builds the hard shape grid then generates softened variants.

    Parameters
    ----------
    resolution      : float  — voxel edge in Å (default 1.0)
    padding         : float  — Å around molecule (default 8.0)
    sigma_list      : list   — Gaussian σ values in Å to apply (default [0.5, 1.0, 2.0])
    rescale_surface : bool   — keep surface voxels exactly +1 after blurring (default True)
    """

    def __init__(
        self,
        resolution:      float = 1.0,
        padding:         float = 8.0,
        sigma_list:      list  = None,
        rescale_surface: bool  = True,
    ):
        self.resolution      = resolution
        self.padding         = padding
        self.sigma_list      = sigma_list or [0.5, 1.0, 2.0]
        self.rescale_surface = rescale_surface
        self._base_builder   = GridBuilder(resolution=resolution, padding=padding)

    def build(self, structure: Structure, mol_type: str) -> SoftGrid:
        mol_grid: MolGrid = self._base_builder.build(structure, mol_type=mol_type)
        hard = mol_grid.shape_grid.copy()

        soft_grids = {}
        for sigma_a in self.sigma_list:
            sigma_vox = sigma_a / self.resolution
            soft = self._apply_gaussian_softening(hard, sigma_vox)
            soft_grids[sigma_a] = soft

        return SoftGrid(
            pdb_id     = structure.pdb_id,
            mol_type   = mol_type,
            origin     = mol_grid.origin,
            resolution = self.resolution,
            hard_grid  = hard,
            soft_grids = soft_grids,
        )

    # ── Core blurring logic ─────────────────────────────────────────────────

    def _apply_gaussian_softening(
        self,
        hard_grid:  np.ndarray,
        sigma_vox:  float,
    ) -> np.ndarray:
        """
        Apply Gaussian blurring to the hard grid, then post-process:

        1.  Interior voxels (−15) are blurred outward, creating a gradient
            clash zone (from −15 deep inside → ~−3 near the surface boundary).
        2.  Surface voxels (+1) are preserved (or re-normalised to +1) so
            surface-contact scores are not diluted.
        3.  Exterior voxels remain 0 (bulk solvent is unchanged).

        The net effect: a 1 Å side-chain overlap that would give −15 in the
        hard grid now gives ~−2 to ~−5 depending on σ.
        """
        interior_mask = hard_grid < 0
        surface_mask  = hard_grid > 0

        # Blur the FULL grid (surface and interior together)
        blurred = gaussian_filter(hard_grid.astype(np.float64), sigma=sigma_vox)

        # Post-process: ensure no voxel INCREASES in positivity beyond +1
        blurred = np.clip(blurred, hard_grid.min(), 1.0)

        # Optionally restore surface voxels to exactly +1
        if self.rescale_surface:
            blurred[surface_mask] = 1.0

        return blurred.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ════════════════════════════════════════════════════════════════════════════

def _middle_slice(grid: np.ndarray, axis: int = 2) -> np.ndarray:
    """Return the central 2D slice along the given axis."""
    mid = grid.shape[axis] // 2
    if axis == 0: return grid[mid, :, :]
    if axis == 1: return grid[:, mid, :]
    return grid[:, :, mid]


def visualize_slice_comparison(sg: SoftGrid, axis: int = 2, cmap: str = "RdBu_r"):
    """
    2D cross-section heatmaps: hard grid vs. each σ.
    Shows the penalty distribution on a representative interior slice.
    Faster and clearer than 3D for understanding the blurring effect.
    """
    sigmas  = list(sg.soft_grids.keys())
    n_plots = 1 + len(sigmas)

    fig = make_subplots(
        rows=1, cols=n_plots,
        subplot_titles=(
            ["Hard (σ=0)"] + [f"Soft σ={s:.1f} Å" for s in sigmas]
        ),
        shared_yaxes=True,
    )

    grids  = [sg.hard_grid] + [sg.soft_grids[s] for s in sigmas]
    labels = ["Hard (σ=0)"] + [f"Soft σ={s:.1f} Å" for s in sigmas]
    global_min = min(g.min() for g in grids)
    global_max = max(g.max() for g in grids)

    axis_labels = {0: "YZ-plane", 1: "XZ-plane", 2: "XY-plane"}

    for col_idx, (grid, lbl) in enumerate(zip(grids, labels), start=1):
        slc = _middle_slice(grid, axis=axis)

        fig.add_trace(
            go.Heatmap(
                z=slc.T,
                colorscale=cmap,
                zmin=global_min,
                zmax=global_max,
                showscale=(col_idx == n_plots),
                colorbar=dict(title="Voxel value"),
            ),
            row=1, col=col_idx,
        )

    axis_name = axis_labels.get(axis, "")
    fig.update_layout(
        title=(
            f"Gaussian Softening — {axis_name} cross-section  |  "
            f"{sg.pdb_id}  ({sg.mol_type})"
        ),
        height=500,
        width=320 * n_plots,
    )
    fig.show()


def visualize_voxel_histogram(sg: SoftGrid):
    """
    Overlaid histograms of voxel-value distributions for the hard grid and
    each softened variant.  Demonstrates how −15 mass is redistributed.
    """
    bins = np.linspace(-16, 2, 90)

    fig = go.Figure()

    all_grids = {"Hard (σ=0)": sg.hard_grid}
    all_grids.update({f"Soft σ={s:.1f} Å": sg.soft_grids[s] for s in sg.soft_grids})

    colours = ["royalblue", "tomato", "darkorange", "mediumpurple"]

    for (label, grid), col in zip(all_grids.items(), colours):
        vals = grid.ravel()
        hist, edges = np.histogram(vals[vals != 0], bins=bins)   # exclude exterior zeros
        mid = 0.5 * (edges[:-1] + edges[1:])

        fig.add_trace(go.Bar(
            x=mid, y=hist,
            name=label,
            marker_color=col,
            opacity=0.6,
            width=(bins[1] - bins[0]) * 0.9,
        ))

    fig.update_layout(
        barmode="overlay",
        title=f"Voxel Value Distribution (non-zero voxels)  |  {sg.pdb_id}  ({sg.mol_type})",
        xaxis_title="Voxel value",
        yaxis_title="Count",
        legend=dict(x=0.01, y=0.99),
        width=900, height=550,
    )
    fig.show()


def visualize_soft_surface_3d(sg: SoftGrid, sigma: float = None,
                               max_pts: int = 50_000):
    """
    3D rendering of the selected soft grid.  Interior voxels are coloured by
    their soft penalty (gradient from blue [deep, severe] to yellow [shallow]),
    surface voxels in red.  Protein side-chain breathing zone visible as the
    yellow-green shell around the interior.

    sigma : which soft grid to render (default: middle sigma)
    """
    if sigma is None:
        sigma = sorted(sg.soft_grids.keys())[len(sg.soft_grids)//2]

    grid = sg.soft_grids.get(sigma, sg.hard_grid)

    surf_idx = np.argwhere(grid > 0)
    int_idx  = np.argwhere(grid < -0.5)

    rng = np.random.default_rng(0)
    if len(surf_idx) > max_pts:
        surf_idx = surf_idx[rng.choice(len(surf_idx), max_pts, replace=False)]
    if len(int_idx) > max_pts:
        int_idx  = int_idx [rng.choice(len(int_idx),  max_pts, replace=False)]

    surf_coords = sg.origin + surf_idx * sg.resolution
    int_coords  = sg.origin + int_idx  * sg.resolution
    int_vals    = grid[int_idx[:,0], int_idx[:,1], int_idx[:,2]]

    fig = go.Figure()

    if len(int_coords):
        fig.add_trace(go.Scatter3d(
            x=int_coords[:,0], y=int_coords[:,1], z=int_coords[:,2],
            mode="markers",
            marker=dict(
                size=3,
                color=int_vals,
                colorscale="plasma",
                cmin=grid.min(), cmax=-0.5,
                colorbar=dict(title="Soft penalty"),
                opacity=0.5,
            ),
            name=f"Interior — soft penalty  (σ={sigma:.1f} Å)",
        ))

    if len(surf_coords):
        fig.add_trace(go.Scatter3d(
            x=surf_coords[:,0], y=surf_coords[:,1], z=surf_coords[:,2],
            mode="markers",
            marker=dict(size=3, color="tomato", opacity=0.6),
            name="Surface (+1)",
        ))

    fig.update_layout(
        title=(
            f"Soft Grid 3D  (σ={sigma:.1f} Å)  |  "
            f"{sg.pdb_id}  ({sg.mol_type})"
        ),
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
        width=1000, height=800,
    )
    fig.show()


def visualize_penalty_gradient_profile(sg: SoftGrid):
    """
    1D penalty profile: distance from the nearest surface voxel vs. penalty.
    Illustrates how the penalty ramps up from ~−2 (shallow clash) to −15
    (deep clash) for each σ.  The hard grid shows a vertical step; soft grids
    show a smooth gradient.
    """
    from scipy.ndimage import distance_transform_edt

    surface_mask = sg.hard_grid > 0
    interior_mask = sg.hard_grid < 0

    dist_from_surface = distance_transform_edt(~surface_mask) * sg.resolution  # Å

    int_dists = dist_from_surface[interior_mask]   # distances inside molecule

    sort_idx = np.argsort(int_dists)
    x = int_dists[sort_idx]

    fig = go.Figure()

    for label, grid in {"Hard (σ=0)": sg.hard_grid, **{f"Soft σ={s:.1f} Å": sg.soft_grids[s] for s in sg.soft_grids}}.items():
        y_vals = grid[interior_mask][sort_idx]

        # Bin for legibility
        bins = np.arange(0, x.max() + 0.5, 0.5)
        bin_means_x, bin_means_y = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (x >= lo) & (x < hi)
            if mask.sum() > 0:
                bin_means_x.append((lo + hi) / 2)
                bin_means_y.append(float(y_vals[mask].mean()))

        fig.add_trace(go.Scatter(
            x=bin_means_x, y=bin_means_y,
            mode="lines+markers",
            name=label,
            line=dict(width=2),
        ))

    fig.update_layout(
        title=f"Penalty vs. Depth from Surface  |  {sg.pdb_id}  ({sg.mol_type})",
        xaxis_title="Distance from molecular surface (Å)",
        yaxis_title="Voxel penalty value",
        width=900, height=500,
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
    print("  Advanced Visualization — Module 3: Soft Grids")
    print(f"  Gaussian Blurring & Conformational Flexibility Simulation")
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

    print(f"\n  Loading cases …")
    cases, _ = load_uu_cases(json_path, pdb_root)
    target = next((c for c in cases if c.complex_id == complex_id), None)

    if target is None:
        print(f"\n  ✗ Complex '{complex_id}' not found. Check ID and paths.")
        return

    print(f"  ✓ Loaded complex {complex_id}\n")

    builder = SoftGridBuilder(
        resolution=1.0,
        padding=8.0,
        sigma_list=[0.5, 1.0, 2.0],
    )

    print("  Building soft grids for protein …")
    pro_sg = builder.build(target.protein_struct, mol_type="protein")
    print(pro_sg.summary())

    print("\n  Building soft grids for RNA …")
    rna_sg = builder.build(target.rna_struct, mol_type="rna")
    print(rna_sg.summary())

    print(f"\n{SEP}")
    print("  Visualization Menu")
    print("  ─────────────────")
    print("  [1]  2D slice comparison — protein  (hard vs. σ=0.5/1.0/2.0 Å)")
    print("  [2]  2D slice comparison — RNA")
    print("  [3]  Voxel histogram — protein  (value distribution)")
    print("  [4]  Voxel histogram — RNA")
    print("  [5]  3D soft grid — protein  (σ=1.0 Å)")
    print("  [6]  3D soft grid — RNA  (σ=1.0 Å)")
    print("  [7]  Penalty gradient profile — protein  (depth vs. penalty)")
    print("  [8]  Penalty gradient profile — RNA")
    print("  [A]  All of the above (8 browser tabs)")
    print(f"{SEP}")

    choice = input("  Enter option(s) (e.g.  1 3 5  or  A): ").strip().upper()

    dispatch = {
        "1": lambda: visualize_slice_comparison(pro_sg),
        "2": lambda: visualize_slice_comparison(rna_sg),
        "3": lambda: visualize_voxel_histogram(pro_sg),
        "4": lambda: visualize_voxel_histogram(rna_sg),
        "5": lambda: visualize_soft_surface_3d(pro_sg, sigma=1.0),
        "6": lambda: visualize_soft_surface_3d(rna_sg, sigma=1.0),
        "7": lambda: visualize_penalty_gradient_profile(pro_sg),
        "8": lambda: visualize_penalty_gradient_profile(rna_sg),
    }

    keys = list(dispatch.keys()) if "A" in choice else choice.split()
    valid = [k for k in keys if k in dispatch]

    if not valid:
        print("  No valid options. Exiting.")
        return

    print(f"\n  Opening {len(valid)} visualisation(s) …")
    for k in valid:
        print(f"    → Option {k}")
        dispatch[k]()

    print(f"\n{SEP}")
    print("  Done.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
