"""
Advanced Visualization Module 4 — Spherical Polar Fourier (SPF) Transforms  (adv_spf.py)
==========================================================================================
Demonstrates the Spherical Polar Fourier framework used by algorithms such as
HEX (Ritchie & Kemp 2000) and FRODOCK (Garzon et al. 2009) to collapse the
full 6-DOF rigid-body search into a single 1-D correlation over SO(3), replacing
the nested "rotate grid → FFT → repeat" loop of phase3/phase4.

WHY SPF?
--------
The standard pipeline (phase4.py) performs K rotations of a Cartesian voxel grid.
Each rotation involves:
    1. A 3D affine_transform (expensive, O(N³))
    2. A 3D FFT       (fast, O(N³ log N))
Total: K × O(N³ log N),  K ≈ 512 at 30°

The SPF approach instead expands each molecule's shape/charge density once into
a set of 3D spherical harmonic coefficients {f_{nlm}}.  The cross-correlation
over ALL rotations simultaneously becomes a sum over (n, l) terms — no grid
rotation required.  Wall-clock speedup vs the Cartesian approach: ~10–100×.

WHAT THIS MODULE PROVIDES (Visualization focus)
------------------------------------------------
Since a full production SPF docking engine would be its own multi-thousand-line
project, this module provides:

    1.  SPF EXPANSION   — expands both protein and RNA onto spherical shells,
                          computes coefficients up to l_max, and stores them.
    2.  POWER SPECTRUM  — |f_{nlm}|² per (n, l) shell; shows which angular
                          frequencies dominate the molecular shape/charge.
    3.  RADIAL PROFILES — f_{n00}(r) for successive shells; intuition for
                          the radial structure of each molecule.
    4.  RECONSTRUCTION  — inverse SPF reconstruction at several l_max cutoffs,
                          visualising the approximation quality (like MRI k-space
                          truncation but for molecules).
    5.  OVERLAP HEATMAP — the theoretical correlation kernel C_{ll'}^n over
                          the SO(3) rotation space, giving a spatial "fingerprint"
                          of how well this pair can dock.
    6.  COMPARATIVE     — side-by-side SPF power spectra for protein vs. RNA,
                          highlighting complementary angular-frequency bands.

Mathematical Background
-----------------------
A function f(r, θ, φ) defined on 3D space is expanded as:

    f(r, θ, φ) = Σ_{n,l,m}  c_{nlm} · R_n(r) · Y_l^m(θ, φ)

where:
    R_n(r)     — radial basis functions (spherical Bessel functions j_l(k_n r),
                  evaluated on a uniform radial grid)
    Y_l^m(θ,φ) — real spherical harmonics
    l          — angular order (0 = spherically symmetric, 1 = dipolar, …)
    m          — projection index, −l ≤ m ≤ l
    n          — radial shell index, 1 ≤ n ≤ N_shells

The rotational cross-correlation:

    C(R) = Σ_{n,l}  Σ_{m,m'}  c^P_{nlm}* · D^l_{mm'}(R) · c^R_{nlm'}

where D^l_{mm'}(R) are the Wigner D-matrix elements, can be evaluated for ALL
rotations R simultaneously via 1D FFT over SO(3) Euler angles.

Standalone usage
----------------
    python adv_spf.py

Dependencies
------------
    pip install numpy scipy plotly pydantic
    # phase1.py  and  phase2.py  must be present in the same folder.
"""

import sys
import math
import numpy as np
from scipy.special import sph_harm, spherical_jn
from scipy.ndimage import map_coordinates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple

sys.path.insert(0, ".")
from phase1 import load_uu_cases, Structure
from phase2 import GridBuilder, MolGrid


# ════════════════════════════════════════════════════════════════════════════
# SPFExpansion — data container
# ════════════════════════════════════════════════════════════════════════════

class SPFExpansion:
    """
    Stores the Spherical Polar Fourier coefficients c_{nlm} for one molecule.

    Attributes
    ----------
    pdb_id        : str
    mol_type      : str
    l_max         : int     maximum angular order
    n_shells      : int     number of radial shells
    center        : np.ndarray (3,)   geometric centre of the molecule in Å
    r_max         : float             outer radius of the expansion sphere in Å
    coeffs        : np.ndarray (n_shells, l_max+1, 2*l_max+1)  complex128
                    coeffs[n, l, m+l_max] = c_{nlm}
    power_spectrum: np.ndarray (n_shells, l_max+1)
                    P[n, l] = Σ_m |c_{nlm}|²  (rotationally invariant)
    """

    def __init__(
        self,
        pdb_id:        str,
        mol_type:      str,
        l_max:         int,
        n_shells:      int,
        center:        np.ndarray,
        r_max:         float,
        coeffs:        np.ndarray,
        power_spectrum: np.ndarray,
    ):
        self.pdb_id         = pdb_id
        self.mol_type       = mol_type
        self.l_max          = l_max
        self.n_shells       = n_shells
        self.center         = center
        self.r_max          = r_max
        self.coeffs         = coeffs
        self.power_spectrum = power_spectrum

    def summary(self) -> str:
        total_coeffs = self.n_shells * (self.l_max + 1)**2
        dominant_l   = int(np.argmax(self.power_spectrum.sum(axis=0)))
        return (
            f"\n{'═'*60}\n"
            f"  SPFExpansion  pdb={self.pdb_id!r}  type={self.mol_type}\n"
            f"{'─'*60}\n"
            f"  l_max          : {self.l_max}\n"
            f"  Radial shells  : {self.n_shells}\n"
            f"  Outer radius   : {self.r_max:.1f} Å\n"
            f"  Total coeffs   : {total_coeffs:,}\n"
            f"  Dominant ang.  : l = {dominant_l}  "
            f"({'monopole' if dominant_l==0 else 'dipole' if dominant_l==1 else 'quadrupole' if dominant_l==2 else f'l={dominant_l}'})\n"
            f"{'═'*60}"
        )


# ════════════════════════════════════════════════════════════════════════════
# SPF Expander
# ════════════════════════════════════════════════════════════════════════════

class SPFExpander:
    """
    Expands a MolGrid onto spherical polar Fourier basis functions.

    Parameters
    ----------
    l_max     : int    maximum angular order (default 10; higher = more detail)
    n_shells  : int    radial shell count    (default 12)
    n_theta   : int    polar integration points  (default 32)
    n_phi     : int    azimuthal integration points (default 64)
    """

    def __init__(
        self,
        l_max:   int = 10,
        n_shells: int = 12,
        n_theta: int = 32,
        n_phi:   int = 64,
    ):
        self.l_max   = l_max
        self.n_shells = n_shells
        self.n_theta = n_theta
        self.n_phi   = n_phi

        # Pre-compute Gauss-Legendre quadrature points for θ ∈ [0, π]
        self._gl_nodes, self._gl_weights = np.polynomial.legendre.leggauss(n_theta)
        self._theta_pts = np.arccos(-self._gl_nodes)   # map [-1,1] → [0,π]

        # Azimuthal points (uniform)
        self._phi_pts = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

        # Pre-compute spherical harmonic table  Y[l, m+l_max, theta_idx, phi_idx]
        print(f"  [SPF] Pre-computing Y_lm table  (l_max={l_max}, "
              f"{n_theta}×{n_phi} angular grid) …", end=" ", flush=True)
        self._ylm = self._precompute_ylm()
        print("done.")

    # ── Public API ───────────────────────────────────────────────────────────

    def expand(self, mol_grid: MolGrid) -> SPFExpansion:
        """
        Expand the grid's shape function onto the SPF basis.
        Uses bilinear interpolation to sample the voxel grid in spherical coords.
        """
        grid   = mol_grid.shape_grid.astype(np.float64)
        origin = mol_grid.origin
        res    = mol_grid.resolution

        center_vox = (np.array(grid.shape) / 2.0)
        center_ang = origin + center_vox * res       # Å

        # Outer radius: half the smallest grid dimension minus a small buffer
        r_max = (min(grid.shape) * res / 2.0) * 0.90

        # Radial shell radii (uniform spacing from 0.5 Å to r_max)
        r_shells = np.linspace(r_max / self.n_shells, r_max, self.n_shells)

        # Coefficients array: [n_shells, l_max+1, 2*l_max+1]
        coeffs = np.zeros((self.n_shells, self.l_max + 1, 2 * self.l_max + 1),
                          dtype=np.complex128)

        # Pre-compute angular integration grid coords for all shells at once
        # and sample the grid
        for n_idx, r in enumerate(r_shells):
            coeffs[n_idx] = self._expand_shell(grid, origin, res, center_ang, r)

        power_spectrum = np.sum(np.abs(coeffs)**2, axis=2)   # (n_shells, l_max+1)

        return SPFExpansion(
            pdb_id         = mol_grid.pdb_id,
            mol_type       = mol_grid.mol_type,
            l_max          = self.l_max,
            n_shells       = self.n_shells,
            center         = center_ang,
            r_max          = r_max,
            coeffs         = coeffs,
            power_spectrum = power_spectrum,
        )

    # ── Internal methods ──────────────────────────────────────────────────────

    def _precompute_ylm(self) -> np.ndarray:
        """
        Returns real-valued Y_lm table  shape: (l_max+1, 2*l_max+1, n_theta, n_phi).
        Uses scipy.special.sph_harm (complex), extracts real part for m≥0
        and imaginary part for m<0  (real spherical harmonics convention).
        """
        L  = self.l_max
        NT = self.n_theta
        NP = self.n_phi

        ylm = np.zeros((L + 1, 2 * L + 1, NT, NP), dtype=np.float64)

        theta_grid = self._theta_pts[:, None]   # (NT, 1)
        phi_grid   = self._phi_pts[None, :]     # (1, NP)

        for l in range(L + 1):
            for m in range(-l, l + 1):
                Y = sph_harm(abs(m), l, phi_grid, theta_grid)  # complex, (NT, NP)
                if m > 0:
                    ylm[l, m + L] = (np.sqrt(2) * Y.real)
                elif m < 0:
                    ylm[l, m + L] = (np.sqrt(2) * Y.imag)
                else:
                    ylm[l, L] = Y.real

        return ylm

    def _expand_shell(
        self,
        grid:       np.ndarray,
        origin:     np.ndarray,
        res:        float,
        center:     np.ndarray,
        r:          float,
    ) -> np.ndarray:
        """
        Compute coefficients c_{lm} for one radial shell of radius r.

        c_{lm} = ∫ f(r, θ, φ) · Y_lm(θ, φ) · sin(θ) dθ dφ
               ≈ Σ_{i,j}  f(r, θ_i, φ_j) · Y_lm(θ_i, φ_j) · w_i · Δφ
        """
        L  = self.l_max
        NT = self.n_theta
        NP = self.n_phi

        # Compute Cartesian coords of every (θ, φ) point on this shell
        sin_theta = np.sin(self._theta_pts)   # (NT,)
        x = center[0] + r * (sin_theta[:, None] * np.cos(self._phi_pts[None, :]))
        y = center[1] + r * (sin_theta[:, None] * np.sin(self._phi_pts[None, :]))
        z = center[2] + r * np.cos(self._theta_pts)[:, None]

        # Convert Å coords to fractional voxel indices
        fx = (x - origin[0]) / res
        fy = (y - origin[1]) / res
        fz = (z - origin[2]) / res

        # Clip to grid bounds to avoid map_coordinates extrapolating
        fx = np.clip(fx, 0, grid.shape[0] - 1)
        fy = np.clip(fy, 0, grid.shape[1] - 1)
        fz = np.clip(fz, 0, grid.shape[2] - 1)

        coords = np.array([fx.ravel(), fy.ravel(), fz.ravel()])
        f_vals = map_coordinates(grid, coords, order=1, mode="nearest").reshape(NT, NP)

        # Integration weights: GL weights × sin(θ) × Δφ
        d_phi   = 2 * np.pi / NP
        weights = (self._gl_weights * sin_theta)[:, None] * d_phi   # (NT, 1)

        # Integrate:  c_{lm} = Σ_{i,j}  f_{ij} · Y_lm_{ij} · w_{i} · dφ
        c = np.zeros((L + 1, 2 * L + 1), dtype=np.complex128)
        weighted_f = f_vals * weights   # (NT, NP)

        for l in range(L + 1):
            for m in range(-l, l + 1):
                Y = self._ylm[l, m + L]       # (NT, NP) real
                c[l, m + L] = np.sum(weighted_f * Y)

        return c

    def reconstruct(self, spf: SPFExpansion, l_max_trunc: int) -> np.ndarray:
        """
        Reconstruct a 3D function on a regular angular grid from SPF coefficients,
        truncated at l_max_trunc.  Returns (n_shells, n_theta, n_phi) array.
        Useful for visualising how much angular detail is captured at each l_max.
        """
        L_use = min(l_max_trunc, spf.l_max)
        out   = np.zeros((spf.n_shells, self.n_theta, self.n_phi))

        for n in range(spf.n_shells):
            for l in range(L_use + 1):
                for m in range(-l, l + 1):
                    c = spf.coeffs[n, l, m + spf.l_max]
                    Y = self._ylm[l, m + spf.l_max]    # (n_theta, n_phi)
                    out[n] += c.real * Y

        return out

    def compute_overlap_kernel(
        self,
        spf_pro: SPFExpansion,
        spf_rna: SPFExpansion,
    ) -> np.ndarray:
        """
        Compute the rotationally-invariant overlap kernel:

            K[n, l] = Σ_m  c^P_{nlm}* · c^R_{nlm}

        This is the part of the SPF cross-correlation that can be summed
        before the SO(3) integral.  Its magnitude tells you which (shell, angular)
        modes contribute most to binding.

        Returns
        -------
        kernel : np.ndarray (n_shells, l_max+1)   complex128
        """
        n = min(spf_pro.n_shells, spf_rna.n_shells)
        l = min(spf_pro.l_max,    spf_rna.l_max)

        kernel = np.zeros((n, l + 1), dtype=np.complex128)
        for ni in range(n):
            for li in range(l + 1):
                cp = spf_pro.coeffs[ni, li, :]      # (2*l_max+1,)
                cr = spf_rna.coeffs[ni, li, :]
                kernel[ni, li] = np.dot(cp.conj(), cr)

        return kernel


# ════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ════════════════════════════════════════════════════════════════════════════

def visualize_power_spectrum(spf: SPFExpansion):
    """
    2D heatmap of power P[n, l] = Σ_m |c_{nlm}|².
    Rows = radial shell index n, columns = angular order l.
    Colour intensity shows which (shell, angular-frequency) modes carry the most
    structural information.
    """
    P = spf.power_spectrum   # (n_shells, l_max+1)

    fig = go.Figure(go.Heatmap(
        z=P,
        x=list(range(spf.l_max + 1)),
        y=[f"n={i+1}" for i in range(spf.n_shells)],
        colorscale="Viridis",
        colorbar=dict(title="Power  |c_{nlm}|²"),
    ))
    fig.update_layout(
        title=f"SPF Power Spectrum  |  {spf.pdb_id}  ({spf.mol_type})",
        xaxis_title="Angular order  l",
        yaxis_title="Radial shell  n",
        width=800, height=500,
    )
    fig.show()


def visualize_radial_profiles(spf: SPFExpansion):
    """
    Plot the monopole (l=0, m=0) coefficient vs. radial shell index.
    This is the spherically-averaged density at each radius — the "radial profile"
    of the molecule.  Also shows l=1 (dipole) and l=2 (quadrupole) for comparison.
    """
    shells = np.arange(1, spf.n_shells + 1)
    r_vals = np.linspace(spf.r_max / spf.n_shells, spf.r_max, spf.n_shells)

    fig = go.Figure()
    for l_show, label, colour in [
        (0, "l=0  (monopole / avg density)", "royalblue"),
        (1, "l=1  (dipole)",                 "tomato"),
        (2, "l=2  (quadrupole)",             "darkorange"),
        (3, "l=3  (octupole)",               "mediumpurple"),
    ]:
        if l_show > spf.l_max:
            continue
        # Use the m=0 component for a single representative curve
        c_vals = spf.coeffs[:, l_show, l_show].real   # m=0 → index l_show in the padded array
        fig.add_trace(go.Scatter(
            x=r_vals, y=c_vals,
            mode="lines+markers", name=label,
            line=dict(color=colour, width=2),
        ))

    fig.add_vline(x=spf.r_max, line_dash="dash", line_color="grey",
                  annotation_text="r_max", annotation_position="top right")
    fig.update_layout(
        title=f"SPF Radial Profiles  (m=0 components)  |  {spf.pdb_id}  ({spf.mol_type})",
        xaxis_title="Radius  r  (Å)",
        yaxis_title="Coefficient value  c_{nl0}",
        width=900, height=500,
    )
    fig.show()


def visualize_reconstruction_comparison(spf: SPFExpansion, expander: SPFExpander,
                                         shell_idx: int = None):
    """
    Show angular reconstruction quality on a representative radial shell.
    Plots the reconstructed angular density as a 2D Mercator-like heatmap
    for l_max_trunc ∈ {2, 5, full}.
    """
    if shell_idx is None:
        shell_idx = spf.n_shells // 2     # middle shell

    l_truncs = [min(2, spf.l_max), min(5, spf.l_max), spf.l_max]
    labels   = [f"l_max = {t}" for t in l_truncs]

    fig = make_subplots(
        rows=1, cols=len(l_truncs),
        subplot_titles=labels,
        shared_yaxes=True,
    )

    phi_deg   = np.degrees(expander._phi_pts)
    theta_deg = np.degrees(expander._theta_pts)

    global_min, global_max = None, None
    recons = []
    for lt in l_truncs:
        rec   = expander.reconstruct(spf, lt)    # (n_shells, n_theta, n_phi)
        slc   = rec[shell_idx]                   # (n_theta, n_phi)
        recons.append(slc)
        if global_min is None:
            global_min, global_max = slc.min(), slc.max()
        else:
            global_min = min(global_min, slc.min())
            global_max = max(global_max, slc.max())

    for col, (slc, lbl) in enumerate(zip(recons, labels), start=1):
        fig.add_trace(
            go.Heatmap(
                z=slc,
                x=phi_deg, y=theta_deg,
                colorscale="RdBu_r",
                zmin=global_min, zmax=global_max,
                showscale=(col == len(l_truncs)),
                colorbar=dict(title="Value"),
            ),
            row=1, col=col,
        )

    r_val = np.linspace(spf.r_max / spf.n_shells, spf.r_max, spf.n_shells)[shell_idx]
    fig.update_layout(
        title=(
            f"SPF Angular Reconstruction  (shell n={shell_idx+1},  r≈{r_val:.1f} Å)"
            f"  |  {spf.pdb_id}  ({spf.mol_type})"
        ),
        xaxis_title="φ (°)", yaxis_title="θ (°)",
        height=450, width=300 * len(l_truncs),
    )
    fig.show()


def visualize_overlap_kernel(
    spf_pro: SPFExpansion,
    spf_rna: SPFExpansion,
    expander: SPFExpander,
):
    """
    Visualise the SPF overlap kernel K[n, l] = Σ_m c^P*_{nlm} · c^R_{nlm}.
    High |K[n, l]| means shell n, angular order l contributes strongly to
    docking — a "fingerprint" of geometric complementarity in SPF space.
    Also shows the cumulative binding potential summed over shells.
    """
    kernel = expander.compute_overlap_kernel(spf_pro, spf_rna)
    K_abs  = np.abs(kernel)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "SPF Overlap Kernel  |K[n,l]|  (all shells)",
            "Cumulative kernel power  Σ_n |K[n,l]|  vs.  l",
        ],
    )

    # Heatmap
    fig.add_trace(
        go.Heatmap(
            z=K_abs,
            x=list(range(kernel.shape[1])),
            y=[f"n={i+1}" for i in range(kernel.shape[0])],
            colorscale="Hot",
            colorbar=dict(title="|K[n,l]|", x=0.45),
        ),
        row=1, col=1,
    )

    # Cumulative over shells
    cum = K_abs.sum(axis=0)
    fig.add_trace(
        go.Bar(
            x=list(range(len(cum))),
            y=cum,
            marker_color="steelblue",
            name="Σ_n |K|",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Angular order  l", row=1, col=1)
    fig.update_yaxes(title_text="Radial shell  n", row=1, col=1)
    fig.update_xaxes(title_text="Angular order  l", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative kernel power", row=1, col=2)

    fig.update_layout(
        title=f"SPF Overlap Kernel  |  {spf_pro.pdb_id}  protein × RNA",
        width=1100, height=500,
        showlegend=False,
    )
    fig.show()


def visualize_comparative_spectra(spf_pro: SPFExpansion, spf_rna: SPFExpansion):
    """
    Protein vs. RNA power spectra overlaid — shows whether the molecules share
    complementary angular-frequency content (necessary for tight docking).
    """
    P_pro = spf_pro.power_spectrum.sum(axis=0)   # sum over shells → (l_max+1,)
    P_rna = spf_rna.power_spectrum.sum(axis=0)

    l_axis = list(range(min(len(P_pro), len(P_rna))))

    # Normalise for fair comparison
    P_pro_n = P_pro[:len(l_axis)] / (P_pro.max() + 1e-12)
    P_rna_n = P_rna[:len(l_axis)] / (P_rna.max() + 1e-12)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=l_axis, y=P_pro_n, name="Protein", opacity=0.7,
                         marker_color="steelblue"))
    fig.add_trace(go.Bar(x=l_axis, y=P_rna_n, name="RNA",     opacity=0.7,
                         marker_color="tomato"))

    # Pearson correlation between spectra
    corr = float(np.corrcoef(P_pro_n, P_rna_n)[0, 1])

    fig.add_annotation(
        x=0.98, y=0.95, xref="paper", yref="paper",
        text=f"Spectral correlation r = {corr:.3f}",
        showarrow=False,
        bgcolor="lightyellow",
        bordercolor="grey",
        borderwidth=1,
    )

    fig.update_layout(
        barmode="group",
        title=f"Comparative SPF Power Spectra  |  {spf_pro.pdb_id}",
        xaxis_title="Angular order  l",
        yaxis_title="Normalised power  (Σ_n |c_{nlm}|²  /  max)",
        width=900, height=500,
    )
    fig.show()


def visualize_spf_3d_sphere(spf: SPFExpansion, expander: SPFExpander,
                             l_max_trunc: int = None):
    """
    3D surface rendering of the SPF reconstruction on the outermost shell.
    Colour maps the angular density; radius is fixed to r_max.
    Gives an intuitive "molecular shape in angular frequency space" view.
    """
    if l_max_trunc is None:
        l_max_trunc = spf.l_max

    rec = expander.reconstruct(spf, l_max_trunc)   # (n_shells, n_theta, n_phi)
    slc = rec[-1]                                   # outermost shell

    theta = expander._theta_pts   # (n_theta,)
    phi   = expander._phi_pts     # (n_phi,)

    PHI, THETA = np.meshgrid(phi, theta)

    r_mod = spf.r_max * (0.85 + 0.15 * (slc - slc.min()) / (slc.ptp() + 1e-12))

    X = r_mod * np.sin(THETA) * np.cos(PHI) + spf.center[0]
    Y = r_mod * np.sin(THETA) * np.sin(PHI) + spf.center[1]
    Z = r_mod * np.cos(THETA) + spf.center[2]

    fig = go.Figure(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=slc,
        colorscale="RdBu_r",
        colorbar=dict(title="Angular density"),
        opacity=0.85,
    ))
    fig.update_layout(
        title=(
            f"SPF 3D Sphere  (outermost shell, l_max={l_max_trunc})"
            f"  |  {spf.pdb_id}  ({spf.mol_type})"
        ),
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
        width=900, height=800,
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
    print("  Advanced Visualization — Module 4: Spherical Polar Fourier (SPF)")
    print(f"  Shape Expansion  |  Power Spectra  |  Overlap Kernel  |  Reconstruction")
    print(f"{SEP}")
    print("  ⚠  This pipeline processes ONE complex at a time.")
    print()
    print("  Note: SPF expansion at l_max=10 takes ~15–30 s on a typical laptop.")
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

    l_max_str = _prompt("  Angular order l_max (6–15, higher = slower)", "8")
    try:
        l_max = int(l_max_str)
        l_max = max(2, min(l_max, 20))
    except ValueError:
        l_max = 8

    print(f"\n  Loading cases …")
    cases, _ = load_uu_cases(json_path, pdb_root)
    target = next((c for c in cases if c.complex_id == complex_id), None)

    if target is None:
        print(f"\n  ✗ Complex '{complex_id}' not found. Check ID and paths.")
        return

    print(f"  ✓ Loaded complex {complex_id}\n")

    base_builder = GridBuilder(resolution=1.0, padding=8.0)
    print("  Building base shape grids …")
    pro_grid = base_builder.build(target.protein_struct, mol_type="protein")
    rna_grid = base_builder.build(target.rna_struct,     mol_type="rna")
    print(f"    Protein grid: {pro_grid.grid_shape}")
    print(f"    RNA grid    : {rna_grid.grid_shape}")

    expander = SPFExpander(l_max=l_max, n_shells=10, n_theta=24, n_phi=48)

    print("\n  Computing SPF expansion — protein …")
    spf_pro = expander.expand(pro_grid)
    print(spf_pro.summary())

    print("\n  Computing SPF expansion — RNA …")
    spf_rna = expander.expand(rna_grid)
    print(spf_rna.summary())

    print(f"\n{SEP}")
    print("  Visualization Menu")
    print("  ─────────────────")
    print("  [1]  Power spectrum heatmap — protein")
    print("  [2]  Power spectrum heatmap — RNA")
    print("  [3]  Radial profiles (l=0–3 components) — protein")
    print("  [4]  Radial profiles — RNA")
    print("  [5]  Angular reconstruction at l_max = 2 / 5 / full — protein")
    print("  [6]  Angular reconstruction — RNA")
    print("  [7]  SPF overlap kernel  (protein × RNA complementarity)")
    print("  [8]  Comparative power spectra  (protein vs. RNA overlay)")
    print("  [9]  3D sphere rendering — protein")
    print("  [10] 3D sphere rendering — RNA")
    print("  [A]  All of the above")
    print(f"{SEP}")

    choice = input("  Enter option(s) (e.g.  1 3 7  or  A): ").strip().upper()

    dispatch = {
        "1":  lambda: visualize_power_spectrum(spf_pro),
        "2":  lambda: visualize_power_spectrum(spf_rna),
        "3":  lambda: visualize_radial_profiles(spf_pro),
        "4":  lambda: visualize_radial_profiles(spf_rna),
        "5":  lambda: visualize_reconstruction_comparison(spf_pro, expander),
        "6":  lambda: visualize_reconstruction_comparison(spf_rna, expander),
        "7":  lambda: visualize_overlap_kernel(spf_pro, spf_rna, expander),
        "8":  lambda: visualize_comparative_spectra(spf_pro, spf_rna),
        "9":  lambda: visualize_spf_3d_sphere(spf_pro, expander),
        "10": lambda: visualize_spf_3d_sphere(spf_rna, expander),
    }

    all_keys = list(dispatch.keys())
    keys = all_keys if "A" in choice else choice.split()
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
