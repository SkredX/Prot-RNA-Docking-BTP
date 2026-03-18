"""
Advanced Visualization Module 2 — Explicit Ion Density Grids  (adv_ion_grids.py)
==================================================================================
Generates a statistical Mg²⁺ ion-probability voxel grid for an RNA structure
and visualizes how predicted ion sites dynamically alter the RNA interaction
surface before FFT docking.

Background
----------
RNA is a massive polyanion: each phosphate group carries ~−1e charge.
Divalent Mg²⁺ ions neutralize this charge, enabling tight protein binding.
Ignoring their positions causes steric and electrostatic artifacts in docking.

This module:
    1.  Identifies regions of high negative electrostatic potential on the RNA
        grid (proxy for favourable Mg²⁺ binding sites).
    2.  Places "probability voxels" reflecting the statistical likelihood of
        Mg²⁺ occupancy at each location (Debye-Hückel + Manning condensation
        heuristic, NOT a full MD simulation).
    3.  Generates a charge-screened RNA electrostatics grid to visualise how
        Mg²⁺ reduces the repulsive potential felt by the incoming protein.
    4.  Shows the predicted ion cloud overlaid on the RNA surface.

Standalone usage
----------------
    python adv_ion_grids.py

    The script will:
      1.  Prompt for ONE complex ID (e.g. 1ASY).
      2.  Ask for the JSON and PDB root paths.
      3.  Build RNA electrostatic + ion-probability grids.
      4.  Open interactive Plotly visualisations (separate browser tabs).

Dependencies
------------
    pip install numpy scipy plotly pydantic
    # phase1.py  and  adv_channel_grids.py  must be in the same folder.
"""

import sys
import math
import numpy as np
from scipy.ndimage import gaussian_filter, label
import plotly.graph_objects as go

sys.path.insert(0, ".")
from phase1 import load_uu_cases, Structure
from adv_channel_grids import MultiChannelBuilder, MultiChannelGrid


# ════════════════════════════════════════════════════════════════════════════
# Physical / empirical constants
# ════════════════════════════════════════════════════════════════════════════

MG_RADIUS_A   = 0.72   # Å — ionic radius of Mg²⁺ (Bondi 1964)
MG_CHARGE     = 2.0    # +2e
WATER_PROBE_A = 1.4    # Å
MANNING_THRESHOLD_KT = -0.5   # kT/e — minimum potential to attract Mg²⁺


# ════════════════════════════════════════════════════════════════════════════
# IonDensityGrid — data container
# ════════════════════════════════════════════════════════════════════════════

class IonDensityGrid:
    """
    Stores the Mg²⁺ ion-probability grid and the charge-screened electrostatic
    grid for one RNA molecule.

    Attributes
    ----------
    pdb_id          : str
    origin          : np.ndarray (3,)
    resolution      : float
    ion_prob_grid   : np.ndarray (Nx,Ny,Nz)  — probability ∈ [0, 1]
    screened_elec   : np.ndarray (Nx,Ny,Nz)  — Mg²⁺-screened potential
    ion_sites       : list of (x,y,z) in Å   — discrete predicted ion centres
    """

    def __init__(
        self,
        pdb_id: str,
        origin: np.ndarray,
        resolution: float,
        ion_prob_grid: np.ndarray,
        screened_elec: np.ndarray,
        ion_sites: list,
    ):
        self.pdb_id        = pdb_id
        self.origin        = origin
        self.resolution    = resolution
        self.ion_prob_grid = ion_prob_grid
        self.screened_elec = screened_elec
        self.ion_sites     = ion_sites

    def summary(self) -> str:
        Nx, Ny, Nz = self.ion_prob_grid.shape
        n_high = int((self.ion_prob_grid > 0.5).sum())
        return (
            f"\n{'═'*60}\n"
            f"  IonDensityGrid  pdb={self.pdb_id!r}  (RNA)\n"
            f"{'─'*60}\n"
            f"  Grid shape       : ({Nx}, {Ny}, {Nz})\n"
            f"  Ion sites found  : {len(self.ion_sites)}\n"
            f"  High-prob voxels : {n_high:,}  (p > 0.5)\n"
            f"  Screened φ range : [{self.screened_elec.min():.3f}, "
                                   f"{self.screened_elec.max():.3f}] kT/e\n"
            f"{'═'*60}"
        )


# ════════════════════════════════════════════════════════════════════════════
# IonDensityBuilder
# ════════════════════════════════════════════════════════════════════════════

class IonDensityBuilder:
    """
    Builds the Mg²⁺ ion-probability grid from an RNA MultiChannelGrid.

    Algorithm (simplified Manning condensation heuristic)
    ─────────────────────────────────────────────────────
    1.  Identify all surface/shell voxels with electrostatic potential
        φ < MANNING_THRESHOLD (i.e., strongly negative).
    2.  Apply Gaussian smoothing to the negative-potential mask to create
        a continuous probability distribution (the ion cloud).
    3.  Find connected-component local minima in the (−φ) field — these
        become discrete ion-site predictions.
    4.  Build the screened potential: φ_screened = φ + Σ_i  Mg_contribution(i)
        where each placed Mg²⁺ adds a +2e Debye-Hückel term.
    """

    def __init__(
        self,
        debye_length:          float = 4.0,    # Å — tighter than protein, more Mg²⁺-specific
        min_site_separation:   float = 5.0,    # Å — min distance between two Mg²⁺ sites
        gaussian_sigma:        float = 1.5,    # voxels — smoothing for probability field
        max_sites:             int   = 30,     # cap on number of Mg²⁺ predicted
    ):
        self.debye_length        = debye_length
        self.min_site_sep        = min_site_separation
        self.gaussian_sigma      = gaussian_sigma
        self.max_sites           = max_sites

    def build(self, rna_mcg: MultiChannelGrid) -> IonDensityGrid:
        """Build ion-probability and screened-electrostatics grids."""

        elec    = rna_mcg.elec_grid            # (Nx, Ny, Nz)
        shape   = rna_mcg.shape_grid
        origin  = rna_mcg.origin
        res     = rna_mcg.resolution

        # ── Step 1: Mask of accessible strongly-negative voxels ──────────────
        # Only consider the surface and hydration-shell region (not interior)
        accessible = shape >= 0                 # surface (+1) or exterior (0)
        negative   = elec < MANNING_THRESHOLD_KT
        candidate_mask = accessible & negative

        # ── Step 2: Probability field via Gaussian smoothing ─────────────────
        raw_prob = (-elec) * candidate_mask.astype(np.float32)
        raw_prob = np.clip(raw_prob, 0, None)
        smoothed = gaussian_filter(raw_prob, sigma=self.gaussian_sigma)

        # Normalise to [0, 1]
        pmax = smoothed.max()
        ion_prob = (smoothed / pmax).astype(np.float32) if pmax > 1e-8 else smoothed

        # ── Step 3: Discrete ion site detection ──────────────────────────────
        ion_sites = self._find_ion_sites(ion_prob, origin, res)

        # ── Step 4: Build Mg²⁺-screened electrostatics ───────────────────────
        screened = self._screen_electrostatics(
            elec.copy(), ion_sites, origin, res, elec.shape
        )

        return IonDensityGrid(
            pdb_id        = rna_mcg.pdb_id,
            origin        = origin,
            resolution    = res,
            ion_prob_grid = ion_prob,
            screened_elec = screened,
            ion_sites     = ion_sites,
        )

    # ── Ion site detection ────────────────────────────────────────────────────

    def _find_ion_sites(self, ion_prob: np.ndarray, origin: np.ndarray,
                        res: float) -> list:
        """
        Greedily pick ion sites as local maxima in the probability field,
        enforcing a minimum separation of `min_site_sep` Å between sites.
        """
        flat_idx = np.argsort(ion_prob.ravel())[::-1]   # descending
        sites    = []
        sep_vox  = self.min_site_sep / res

        for fi in flat_idx:
            if ion_prob.ravel()[fi] < 0.3:               # probability too low
                break
            ix, iy, iz = np.unravel_index(fi, ion_prob.shape)
            coord = origin + np.array([ix, iy, iz]) * res

            # Enforce separation
            too_close = any(
                np.linalg.norm(coord - np.array(s)) < self.min_site_sep
                for s in sites
            )
            if not too_close:
                sites.append(coord.tolist())
                if len(sites) >= self.max_sites:
                    break

        return sites

    # ── Charge screening ──────────────────────────────────────────────────────

    def _screen_electrostatics(
        self,
        elec:   np.ndarray,
        sites:  list,
        origin: np.ndarray,
        res:    float,
        dims:   tuple,
    ) -> np.ndarray:
        """
        Add +2e Debye-Hückel contribution for each predicted Mg²⁺ site.
        This approximates partial neutralisation of the RNA polyanion.
        """
        Nx, Ny, Nz = dims
        cutoff_vox = int(math.ceil(4.0 * self.debye_length / res))
        eps        = 80.0   # water dielectric for ion in solvent

        for site in sites:
            sx, sy, sz = site
            ix0 = int(round((sx - origin[0]) / res))
            iy0 = int(round((sy - origin[1]) / res))
            iz0 = int(round((sz - origin[2]) / res))

            lo = np.clip([ix0-cutoff_vox, iy0-cutoff_vox, iz0-cutoff_vox], 0, [Nx-1, Ny-1, Nz-1])
            hi = np.clip([ix0+cutoff_vox+1, iy0+cutoff_vox+1, iz0+cutoff_vox+1], 0, [Nx, Ny, Nz])

            ix = np.arange(lo[0], hi[0])
            iy = np.arange(lo[1], hi[1])
            iz = np.arange(lo[2], hi[2])

            dx = (origin[0] + ix*res - sx)[:, None, None]
            dy = (origin[1] + iy*res - sy)[None, :, None]
            dz = (origin[2] + iz*res - sz)[None, None, :]

            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            dist = np.where(dist < 0.5, 0.5, dist)

            phi_mg = (MG_CHARGE * np.exp(-dist / self.debye_length) / (eps * dist)).astype(np.float32)
            elec[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += phi_mg

        return elec


# ════════════════════════════════════════════════════════════════════════════
# Visualisation
# ════════════════════════════════════════════════════════════════════════════

def _downsample(indices: np.ndarray, max_pts: int, seed: int = 42) -> np.ndarray:
    if len(indices) > max_pts:
        return indices[np.random.default_rng(seed).choice(len(indices), max_pts, replace=False)]
    return indices


def visualize_ion_probability(rna_mcg: MultiChannelGrid,
                               ion_grid: IonDensityGrid,
                               max_pts: int = 50_000):
    """
    Show the RNA molecular surface coloured by ion-occupancy probability.
    Predicted Mg²⁺ sites rendered as large green spheres.
    """
    surf_idx = np.argwhere(rna_mcg.shape_grid > 0)
    surf_idx = _downsample(surf_idx, max_pts)

    surf_coords = rna_mcg.origin + surf_idx * rna_mcg.resolution
    prob_vals   = ion_grid.ion_prob_grid[surf_idx[:,0], surf_idx[:,1], surf_idx[:,2]]

    fig = go.Figure()

    # RNA surface coloured by probability
    fig.add_trace(go.Scatter3d(
        x=surf_coords[:,0], y=surf_coords[:,1], z=surf_coords[:,2],
        mode="markers",
        marker=dict(
            size=3,
            color=prob_vals,
            colorscale="Viridis",
            cmin=0.0, cmax=1.0,
            colorbar=dict(title="Mg²⁺ prob."),
            opacity=0.7,
        ),
        name="RNA surface (ion probability)",
    ))

    # Predicted Mg²⁺ site markers
    if ion_grid.ion_sites:
        sites = np.array(ion_grid.ion_sites)
        fig.add_trace(go.Scatter3d(
            x=sites[:,0], y=sites[:,1], z=sites[:,2],
            mode="markers",
            marker=dict(size=10, color="lime", symbol="diamond",
                        line=dict(color="darkgreen", width=2)),
            name=f"Predicted Mg²⁺ sites ({len(ion_grid.ion_sites)})",
        ))

    fig.update_layout(
        title=f"Mg²⁺ Ion Probability Grid  |  {ion_grid.pdb_id}  (RNA)",
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
        width=1000, height=800,
    )
    fig.show()


def visualize_screened_vs_raw_electrostatics(rna_mcg: MultiChannelGrid,
                                              ion_grid: IonDensityGrid,
                                              max_pts: int = 40_000):
    """
    Side-by-side (two browser tabs) comparison of raw vs. Mg²⁺-screened
    electrostatic potential on the RNA surface.
    """
    surf_idx = np.argwhere(rna_mcg.shape_grid > 0)
    surf_idx = _downsample(surf_idx, max_pts)
    coords   = rna_mcg.origin + surf_idx * rna_mcg.resolution

    raw_vals      = rna_mcg.elec_grid[surf_idx[:,0], surf_idx[:,1], surf_idx[:,2]]
    screened_vals = ion_grid.screened_elec[surf_idx[:,0], surf_idx[:,1], surf_idx[:,2]]

    vlo = min(np.percentile(raw_vals, 2), np.percentile(screened_vals, 2))
    vhi = max(np.percentile(raw_vals, 98), np.percentile(screened_vals, 98))

    for vals, label_str in [(raw_vals, "Raw"), (screened_vals, "Mg²⁺-Screened")]:
        fig = go.Figure(go.Scatter3d(
            x=coords[:,0], y=coords[:,1], z=coords[:,2],
            mode="markers",
            marker=dict(
                size=3, color=vals, colorscale="RdBu_r",
                cmin=vlo, cmax=vhi,
                colorbar=dict(title="φ (kT/e)"),
                opacity=0.8,
            ),
            name=label_str,
        ))
        fig.update_layout(
            title=f"Electrostatics — {label_str}  |  {ion_grid.pdb_id}  (RNA)",
            scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
            width=950, height=750,
        )
        fig.show()


def visualize_ion_cloud_3d(rna_mcg: MultiChannelGrid,
                            ion_grid: IonDensityGrid,
                            prob_threshold: float = 0.4,
                            max_pts: int = 40_000):
    """
    Volumetric ion-cloud rendering: all voxels with probability above the
    threshold, colour-coded by probability, plus the RNA surface for context.
    """
    cloud_idx = np.argwhere(ion_grid.ion_prob_grid > prob_threshold)
    surf_idx  = np.argwhere(rna_mcg.shape_grid > 0)

    cloud_idx = _downsample(cloud_idx, max_pts)
    surf_idx  = _downsample(surf_idx, max_pts // 2)

    cloud_coords = rna_mcg.origin + cloud_idx * rna_mcg.resolution
    surf_coords  = rna_mcg.origin + surf_idx  * rna_mcg.resolution
    cloud_vals   = ion_grid.ion_prob_grid[cloud_idx[:,0], cloud_idx[:,1], cloud_idx[:,2]]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=surf_coords[:,0], y=surf_coords[:,1], z=surf_coords[:,2],
        mode="markers",
        marker=dict(size=2, color="steelblue", opacity=0.2),
        name="RNA surface",
    ))

    if len(cloud_coords):
        fig.add_trace(go.Scatter3d(
            x=cloud_coords[:,0], y=cloud_coords[:,1], z=cloud_coords[:,2],
            mode="markers",
            marker=dict(
                size=4,
                color=cloud_vals,
                colorscale="YlOrRd",
                cmin=prob_threshold, cmax=1.0,
                colorbar=dict(title="Mg²⁺ prob."),
                opacity=0.6,
            ),
            name=f"Ion cloud (p > {prob_threshold})",
        ))

    if ion_grid.ion_sites:
        sites = np.array(ion_grid.ion_sites)
        fig.add_trace(go.Scatter3d(
            x=sites[:,0], y=sites[:,1], z=sites[:,2],
            mode="markers",
            marker=dict(size=12, color="lime", symbol="diamond",
                        line=dict(color="darkgreen", width=2)),
            name=f"Mg²⁺ site predictions ({len(ion_grid.ion_sites)})",
        ))

    fig.update_layout(
        title=f"Mg²⁺ Ion Cloud  (p > {prob_threshold})  |  {ion_grid.pdb_id}",
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
    print("  Advanced Visualization — Module 2: Explicit Ion Density Grids")
    print(f"  The Magnesium Problem — Mg²⁺ Probability & Charge Screening")
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
        print(f"\n  ✗ Complex '{complex_id}' not found.  Check ID and paths.")
        return

    print(f"  ✓ Loaded complex {complex_id}\n")

    print("  Building RNA multi-channel grid (electrostatics required) …")
    mc_builder = MultiChannelBuilder(resolution=1.0, padding=8.0)
    rna_mcg = mc_builder.build(target.rna_struct, mol_type="rna")
    print(f"  ✓ RNA multi-channel grid built  (shape {rna_mcg.grid_shape})")

    print("\n  Computing Mg²⁺ ion-probability grid …")
    ion_builder = IonDensityBuilder()
    ion_grid = ion_builder.build(rna_mcg)
    print(ion_grid.summary())

    print(f"\n{SEP}")
    print("  Visualization Menu")
    print("  ─────────────────")
    print("  [1]  RNA surface coloured by Mg²⁺ probability + site markers")
    print("  [2]  Raw vs. Mg²⁺-screened electrostatics (2 tabs)")
    print("  [3]  Volumetric ion-cloud rendering (p > 0.4)")
    print("  [A]  All of the above")
    print(f"{SEP}")

    choice = input("  Enter option(s) (e.g.  1 3  or  A): ").strip().upper()

    dispatch = {
        "1": lambda: visualize_ion_probability(rna_mcg, ion_grid),
        "2": lambda: visualize_screened_vs_raw_electrostatics(rna_mcg, ion_grid),
        "3": lambda: visualize_ion_cloud_3d(rna_mcg, ion_grid),
    }

    keys = ["1", "2", "3"] if "A" in choice else choice.split()
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
