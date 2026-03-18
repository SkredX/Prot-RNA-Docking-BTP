"""
Advanced Visualization Module 5 — 3D-CNN AI Scoring Function  (adv_cnn_scoring.py)
====================================================================================
A self-contained 3D Convolutional Neural Network that rescores the top-N docking
poses produced by the FFT engine (phase4.py), learning non-linear binding motifs
that shape-complementarity alone cannot capture.

PIPELINE OVERVIEW
-----------------
                    ┌─────────────────────────────────────────────┐
    FFT poses       │  For each of the top N poses:               │
    (phase4.py) ──► │  1. Apply R + t to RNA coordinates          │  3-channel
                    │  2. Build voxel grid around the interface    │  voxel tensor
                    │  3. Stack: Shape | Electrostatics | Hydrophob│ ──────────►
                    └─────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  3D-CNN Scorer       │   Architecture:
                    │  (PyTorch)           │   Conv3d(3,32) → BN → ReLU → MaxPool
                    │                      │   Conv3d(32,64) → BN → ReLU → MaxPool
                    │                      │   Conv3d(64,128) → BN → ReLU → AdaptPool
                    │                      │   FC(512) → Dropout → FC(1) → Sigmoid
                    └──────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  Re-ranked pose list  │  + saliency maps showing
                    │  (CNN score ∈ [0,1])  │    WHICH voxels drove the score
                    └──────────────────────┘

TRAINING DATA STRATEGY (described in detail below)
--------------------------------------------------
We cannot train from scratch here — we have no labelled binding data.
This module therefore implements THREE modes:

    MODE A: DEMO (default)  — The network is randomly initialised.  This lets
            you see the entire pipeline, architecture, and visualisations work
            end-to-end.  Scores will be random-noise baselines.  Use this to
            understand the workflow before training.

    MODE B: SELF-SUPERVISED CONTRASTIVE TRAINING on the current complex.
            Positive examples: the top-scoring FFT pose + the native complex
            (if you have the bound PDB).
            Negative examples: randomly perturbed/clashing poses (score < 0 FFT).
            Trains for a small number of epochs in-session (~2–5 min on CPU).
            This gives meaningful relative re-ranking for this one complex.

    MODE C: LOAD SAVED WEIGHTS  — Load a .pt checkpoint previously saved
            by Mode B, enabling fast inference across multiple sessions.

WHAT THIS MODULE PROVIDES
--------------------------
    1.  Network architecture summary with parameter count.
    2.  Per-channel input tensor visualisation (3D slices) for a selected pose.
    3.  Training loss / validation loss curves (Mode B).
    4.  Re-ranked pose table: FFT rank vs. CNN rank, score scatter plot.
    5.  Gradient-weighted class activation maps (Grad-CAM 3D) showing which
        voxels in the binding pocket drove the CNN's score — analogous to
        "attention maps" in NLP.
    6.  Score distribution histogram: CNN scores for top-N vs. random poses.
    7.  Architecture diagram rendered as an annotated flow chart.

Standalone usage
----------------
    python adv_cnn_scoring.py

    You will be prompted for:
        • Complex ID
        • JSON + PDB root paths
        • Mode (A / B / C)
        • Path to results.pkl  (from phase4/run.py)  — optional; if absent,
          the module synthesises plausible poses from SO(3) sampling for demo.

Dependencies
------------
    pip install numpy scipy plotly pydantic torch
    # phase1.py, phase2.py, phase3.py, adv_channel_grids.py  must be present.
"""

import sys
import os
import math
import time
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, ".")
from phase1 import load_uu_cases, Structure
from phase2 import GridBuilder, MolGrid, get_vdw_radius
from phase3 import generate_uniform_rotations
from adv_channel_grids import MultiChannelBuilder, MultiChannelGrid

# ── Torch import (graceful fallback) ──────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════

BOX_SIZE_A  = 24       # Å — cubic box around the interface for CNN input
VOXEL_RES_A = 1.0      # Å/voxel
BOX_VOXELS  = int(BOX_SIZE_A / VOXEL_RES_A)   # 24 voxels per side
N_CHANNELS  = 3        # Shape, Electrostatics, Hydrophobicity


# ════════════════════════════════════════════════════════════════════════════
# Hydrophobicity scale  (Kyte-Doolittle, rescaled to [0, 1])
# ════════════════════════════════════════════════════════════════════════════

_KD_SCALE: dict = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4,"THR": -0.7,"SER": -0.8,
    "TRP": -0.9,"TYR": -1.3,"PRO": -1.6,"HIS": -3.2,"GLU": -3.5,
    "GLN": -3.5,"ASP": -3.5,"ASN": -3.5,"LYS": -3.9,"ARG": -4.5,
}
_KD_MIN, _KD_MAX = -4.5, 4.5

def get_hydrophobicity(res_name: str) -> float:
    """Return normalised hydrophobicity ∈ [0, 1] for a residue (0 = polar, 1 = hydrophobic)."""
    raw = _KD_SCALE.get(res_name, 0.0)
    return (raw - _KD_MIN) / (_KD_MAX - _KD_MIN)


# ════════════════════════════════════════════════════════════════════════════
# Pose container
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Pose:
    """Represents a single docked pose."""
    fft_rank:           int
    fft_score:          float
    rotation_matrix:    np.ndarray    # (3,3)
    translation_vector: np.ndarray    # (3,) Å
    cnn_score:          float = 0.0   # filled in after scoring


# ════════════════════════════════════════════════════════════════════════════
# Interface box extractor
# ════════════════════════════════════════════════════════════════════════════

class InterfaceBoxExtractor:
    """
    For a given docked pose, extract a (N_CHANNELS, BOX_VOXELS³) tensor
    centred at the interface between protein and RNA.

    Channels:
        0 — Shape         (KK encoding on combined complex grid)
        1 — Electrostatics (protein + docked RNA, additive)
        2 — Hydrophobicity (protein surface residues only, per-voxel)
    """

    def __init__(self, resolution: float = VOXEL_RES_A, box_size: int = BOX_VOXELS):
        self.resolution = resolution
        self.box_size   = box_size
        self._mc_builder = MultiChannelBuilder(resolution=resolution, padding=4.0)
        self._shape_builder = GridBuilder(resolution=resolution, padding=4.0)

    def extract(
        self,
        pose:         Pose,
        protein_struct: Structure,
        rna_struct:     Structure,
    ) -> np.ndarray:
        """
        Returns float32 array of shape (N_CHANNELS, box_size, box_size, box_size).
        """
        # Step 1: Apply rotation + translation to RNA coordinates
        rna_coords = self._get_all_coords(rna_struct)
        rna_center = rna_coords.mean(axis=0)
        R = pose.rotation_matrix
        t = pose.translation_vector

        rna_coords_docked = (R @ (rna_coords - rna_center).T).T + rna_center + t

        # Step 2: Find interface center (midpoint of closest protein-RNA atom pair)
        pro_coords = self._get_all_coords(protein_struct)
        interface_center = self._find_interface_center(pro_coords, rna_coords_docked)

        # Step 3: Build box around interface center
        half    = (self.box_size * self.resolution) / 2.0
        box_origin = interface_center - half

        # Build each channel
        shape_ch = self._build_shape_channel(
            protein_struct, rna_struct, rna_coords_docked, box_origin
        )
        elec_ch  = self._build_elec_channel(
            protein_struct, rna_struct, rna_coords_docked, box_origin
        )
        hydro_ch = self._build_hydro_channel(
            protein_struct, box_origin
        )

        tensor = np.stack([shape_ch, elec_ch, hydro_ch], axis=0).astype(np.float32)
        return tensor

    # ── Per-channel builders ─────────────────────────────────────────────────

    def _build_shape_channel(self, pro_struct, rna_struct, rna_coords_docked, box_origin):
        Nx = Ny = Nz = self.box_size
        r = self.resolution
        grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)

        # Protein atoms
        pro_atoms = [a for c in pro_struct.protein_chains() for a in c.atoms]
        self._fill_shape_grid(grid, pro_atoms, box_origin, r)

        # Docked RNA atoms (use shifted coordinates)
        rna_atoms  = [a for c in rna_struct.rna_chains() for a in c.atoms]
        rna_coords = self._get_all_coords(rna_struct)
        rna_center = rna_coords.mean(axis=0)
        R, t = None, None   # not needed — we already have rna_coords_docked

        self._fill_shape_grid_from_coords(grid, rna_atoms, rna_coords_docked, box_origin, r)
        return grid

    def _build_elec_channel(self, pro_struct, rna_struct, rna_coords_docked, box_origin):
        Nx = Ny = Nz = self.box_size
        grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        r = self.resolution

        from adv_channel_grids import get_partial_charge

        # Protein contribution
        pro_atoms = [a for c in pro_struct.protein_chains() for a in c.atoms]
        for atom in pro_atoms:
            q = get_partial_charge(atom)
            if abs(q) < 1e-6: continue
            self._add_coulomb(grid, atom.x, atom.y, atom.z, q, box_origin, r)

        # Docked RNA contribution
        rna_atoms  = [a for c in rna_struct.rna_chains() for a in c.atoms]
        rna_coords = np.array([[a.x, a.y, a.z] for a in rna_atoms])
        for atom, (dx, dy, dz) in zip(rna_atoms, rna_coords_docked):
            q = get_partial_charge(atom)
            if abs(q) < 1e-6: continue
            self._add_coulomb(grid, dx, dy, dz, q, box_origin, r)

        return grid

    def _build_hydro_channel(self, pro_struct, box_origin):
        Nx = Ny = Nz = self.box_size
        grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        r = self.resolution

        pro_atoms = [a for c in pro_struct.protein_chains() for a in c.atoms]
        for atom in pro_atoms:
            hval = get_hydrophobicity(atom.res_name)
            if hval < 0.01: continue

            ix = int(round((atom.x - box_origin[0]) / r))
            iy = int(round((atom.y - box_origin[1]) / r))
            iz = int(round((atom.z - box_origin[2]) / r))

            if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
                grid[ix, iy, iz] = max(grid[ix, iy, iz], hval)

        return grid

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_all_coords(self, struct):
        return np.array([[a.x, a.y, a.z] for c in struct.chains for a in c.atoms])

    def _find_interface_center(self, pro_coords, rna_coords_docked):
        """Return midpoint of the nearest protein-RNA atom pair."""
        from scipy.spatial import cKDTree
        tree   = cKDTree(pro_coords)
        dists, idxs = tree.query(rna_coords_docked, k=1)
        nn_rna_idx = int(np.argmin(dists))
        nn_pro_idx = idxs[nn_rna_idx]
        return (pro_coords[nn_pro_idx] + rna_coords_docked[nn_rna_idx]) / 2.0

    def _fill_shape_grid(self, grid, atoms, origin, r):
        Nx, Ny, Nz = grid.shape
        for atom in atoms:
            radius = get_vdw_radius(atom)
            ix0 = int(round((atom.x - origin[0]) / r))
            iy0 = int(round((atom.y - origin[1]) / r))
            iz0 = int(round((atom.z - origin[2]) / r))
            k = int(math.ceil(radius / r)) + 1
            for ix in range(max(0, ix0-k), min(Nx, ix0+k+1)):
                for iy in range(max(0, iy0-k), min(Ny, iy0+k+1)):
                    for iz in range(max(0, iz0-k), min(Nz, iz0+k+1)):
                        dx = origin[0] + ix*r - atom.x
                        dy = origin[1] + iy*r - atom.y
                        dz = origin[2] + iz*r - atom.z
                        if dx*dx + dy*dy + dz*dz <= radius*radius:
                            grid[ix, iy, iz] = -15.0

    def _fill_shape_grid_from_coords(self, grid, atoms, coords_arr, origin, r):
        Nx, Ny, Nz = grid.shape
        for atom, (ax, ay, az) in zip(atoms, coords_arr):
            radius = get_vdw_radius(atom)
            ix0 = int(round((ax - origin[0]) / r))
            iy0 = int(round((ay - origin[1]) / r))
            iz0 = int(round((az - origin[2]) / r))
            k = int(math.ceil(radius / r)) + 1
            for ix in range(max(0, ix0-k), min(Nx, ix0+k+1)):
                for iy in range(max(0, iy0-k), min(Ny, iy0+k+1)):
                    for iz in range(max(0, iz0-k), min(Nz, iz0+k+1)):
                        dx = origin[0] + ix*r - ax
                        dy = origin[1] + iy*r - ay
                        dz = origin[2] + iz*r - az
                        if dx*dx + dy*dy + dz*dz <= radius*radius:
                            grid[ix, iy, iz] = 1.0   # RNA surface

    def _add_coulomb(self, grid, ax, ay, az, q, origin, r):
        Nx, Ny, Nz = grid.shape
        cutoff  = 10.0   # Å
        k_vox   = int(math.ceil(cutoff / r))
        eps     = 4.0
        lambda_ = 8.0

        ix0 = int(round((ax - origin[0]) / r))
        iy0 = int(round((ay - origin[1]) / r))
        iz0 = int(round((az - origin[2]) / r))

        for ix in range(max(0, ix0-k_vox), min(Nx, ix0+k_vox+1)):
            for iy in range(max(0, iy0-k_vox), min(Ny, iy0+k_vox+1)):
                for iz in range(max(0, iz0-k_vox), min(Nz, iz0+k_vox+1)):
                    dx = origin[0] + ix*r - ax
                    dy = origin[1] + iy*r - ay
                    dz = origin[2] + iz*r - az
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if dist < 0.5: dist = 0.5
                    grid[ix, iy, iz] += q * math.exp(-dist/lambda_) / (eps * dist)


# ════════════════════════════════════════════════════════════════════════════
# 3D CNN Architecture
# ════════════════════════════════════════════════════════════════════════════

def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed.\n"
            "Install it with:  pip install torch\n"
            "or visit https://pytorch.org/get-started/locally/"
        )


class BindingScorer3DCNN(nn.Module):
    """
    3D Convolutional Neural Network for protein-RNA interface scoring.

    Input:  (batch, 3, 24, 24, 24) float32 tensor
            channels: [shape, electrostatics, hydrophobicity]

    Output: (batch, 1) float32, sigmoid activation → score ∈ [0, 1]
            1 = predicted binder, 0 = predicted non-binder

    Architecture
    ─────────────
    Conv3d(3 → 32,  k=3, pad=1) → BatchNorm → ReLU → MaxPool(2)   → 12³
    Conv3d(32 → 64, k=3, pad=1) → BatchNorm → ReLU → MaxPool(2)   → 6³
    Conv3d(64 →128, k=3, pad=1) → BatchNorm → ReLU → AdaptPool(3) → 3³
    Flatten → FC(3456 → 512) → Dropout(0.4) → FC(512 → 1) → Sigmoid
    """

    def __init__(self, in_channels: int = N_CHANNELS, box: int = BOX_VOXELS):
        super().__init__()
        _require_torch()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d(3),
        )
        flat_size = 128 * 3 * 3 * 3   # = 3456
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_str(self) -> str:
        lines = [
            f"BindingScorer3DCNN",
            f"  Input  : (batch, {N_CHANNELS}, {BOX_VOXELS}, {BOX_VOXELS}, {BOX_VOXELS})",
            f"  ─────────────────────────────────────────────────",
            f"  Conv3d (3→32,  k=3) + BN + ReLU + MaxPool → 32×12³",
            f"  Conv3d (32→64, k=3) + BN + ReLU + MaxPool → 64×6³",
            f"  Conv3d (64→128,k=3) + BN + ReLU + AdaptPool→128×3³",
            f"  Flatten  →  3,456",
            f"  FC (3456 → 512)  + ReLU  + Dropout(0.4)",
            f"  FC (512  → 1)    + Sigmoid",
            f"  ─────────────────────────────────────────────────",
            f"  Total parameters : {self.param_count():,}",
            f"  Output : score ∈ [0, 1]  (1 = predicted binder)",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Trainer (Mode B — contrastive self-supervised)
# ════════════════════════════════════════════════════════════════════════════

class CNNTrainer:
    """
    Self-supervised contrastive trainer on the current complex.

    Positive examples  (label=1): top-K FFT poses (assumed near-native)
    Negative examples  (label=0): bottom-K FFT poses (clashing / far from native)

    Trains with Binary Cross-Entropy loss.
    """

    def __init__(
        self,
        model:       "BindingScorer3DCNN",
        device:      "torch.device",
        lr:          float = 1e-3,
        epochs:      int   = 20,
        batch_size:  int   = 4,
    ):
        _require_torch()
        self.model      = model.to(device)
        self.device     = device
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []

    def train(
        self,
        positive_tensors: List[np.ndarray],
        negative_tensors: List[np.ndarray],
    ) -> Tuple[List[float], List[float]]:
        """
        Train the model.  Returns (train_losses, val_losses) per epoch.
        """
        # Build dataset
        X = np.stack(positive_tensors + negative_tensors, axis=0)
        y = np.array([1.0] * len(positive_tensors) + [0.0] * len(negative_tensors),
                     dtype=np.float32)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        n_val = max(1, len(X_t) // 5)
        perm  = torch.randperm(len(X_t))
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]

        X_train, y_train = X_t[train_idx], y_t[train_idx]
        X_val,   y_val   = X_t[val_idx],   y_t[val_idx]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        print(f"  Training: {len(X_train)} positives+negatives  |  "
              f"Val: {len(X_val)}  |  Epochs: {self.epochs}")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            perm2 = torch.randperm(len(X_train))

            for i in range(0, len(X_train), self.batch_size):
                idx_batch = perm2[i:i + self.batch_size]
                xb = X_train[idx_batch].to(self.device)
                yb = y_train[idx_batch].to(self.device)

                optimizer.zero_grad()
                pred  = self.model(xb)
                loss  = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)

            train_loss = epoch_loss / len(X_train)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val.to(self.device))
                val_loss = criterion(val_pred, y_val.to(self.device)).item()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:>3}/{self.epochs}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        return self.train_losses, self.val_losses


# ════════════════════════════════════════════════════════════════════════════
# Grad-CAM 3D
# ════════════════════════════════════════════════════════════════════════════

def compute_gradcam_3d(
    model: "BindingScorer3DCNN",
    tensor: np.ndarray,
    device: "torch.device",
) -> np.ndarray:
    """
    Compute Gradient-weighted Class Activation Map (Grad-CAM) for the last
    convolutional layer.  Returns a (BOX_VOXELS, BOX_VOXELS, BOX_VOXELS)
    saliency map normalised to [0, 1].

    High values → voxels that most positively influenced the score.
    """
    _require_torch()

    model.eval()
    x = torch.tensor(tensor[None], dtype=torch.float32, device=device)

    # Store intermediate activations and gradients
    activations = {}
    gradients   = {}

    def _fwd_hook(module, inp, out):
        activations["last_conv"] = out.detach()

    def _bwd_hook(module, grad_in, grad_out):
        gradients["last_conv"] = grad_out[0].detach()

    # Hook the last Conv3d layer (conv3's Conv3d)
    last_conv = model.conv3[0]
    fwd_h = last_conv.register_forward_hook(_fwd_hook)
    bwd_h = last_conv.register_full_backward_hook(_bwd_hook)

    x.requires_grad_(True)
    score = model(x)         # forward
    score.backward()         # backward

    fwd_h.remove()
    bwd_h.remove()

    act  = activations["last_conv"][0]   # (C, d, h, w)
    grad = gradients["last_conv"][0]     # (C, d, h, w)

    # Global average-pool gradients over spatial dims
    weights = grad.mean(dim=(1, 2, 3))   # (C,)

    cam = (weights[:, None, None, None] * act).sum(dim=0)   # (d, h, w)
    cam = F.relu(cam)
    cam = cam.cpu().numpy()

    # Upsample to box size
    from scipy.ndimage import zoom
    scale = BOX_VOXELS / cam.shape[0]
    cam   = zoom(cam, scale, order=1)
    cam   = cam[:BOX_VOXELS, :BOX_VOXELS, :BOX_VOXELS]

    # Normalise
    vmax = cam.max()
    if vmax > 1e-8:
        cam = cam / vmax

    return cam.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ════════════════════════════════════════════════════════════════════════════

def visualize_architecture(model: "BindingScorer3DCNN"):
    """Render an annotated architecture flowchart using Plotly shapes + text."""
    _require_torch()
    print(model.architecture_str())

    layers = [
        ("Input",        f"(batch, 3, {BOX_VOXELS}³)",  "#4a90d9"),
        ("Conv3d 3→32",  "k=3, BN, ReLU\n+ MaxPool → 12³", "#5cb85c"),
        ("Conv3d 32→64", "k=3, BN, ReLU\n+ MaxPool → 6³",  "#5cb85c"),
        ("Conv3d 64→128","k=3, BN, ReLU\n+ AdaptPool → 3³","#5cb85c"),
        ("Flatten",      "3,456 features",                  "#f0ad4e"),
        ("FC 3456→512",  "ReLU, Dropout(0.4)",              "#d9534f"),
        ("FC 512→1",     "Sigmoid → score ∈ [0,1]",         "#9b59b6"),
    ]

    fig = go.Figure()
    box_w, box_h, gap = 0.3, 0.10, 0.05
    x0 = 0.35

    for i, (name, detail, colour) in enumerate(layers):
        y = 1.0 - i * (box_h + gap)
        fig.add_shape(type="rect",
                      x0=x0, y0=y-box_h, x1=x0+box_w, y1=y,
                      fillcolor=colour, opacity=0.85,
                      line=dict(color="white", width=1.5))
        fig.add_annotation(x=x0+box_w/2, y=y-box_h/2,
                           text=f"<b>{name}</b><br><sub>{detail}</sub>",
                           showarrow=False, font=dict(size=11, color="white"),
                           align="center")
        if i < len(layers) - 1:
            y_arr = y - box_h
            fig.add_annotation(x=x0+box_w/2, y=y_arr-gap/2,
                               text="▼", showarrow=False,
                               font=dict(size=16, color="#aaa"))

    fig.update_layout(
        title=f"3D-CNN Architecture  |  {model.param_count():,} parameters",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[-0.1, 1.1]),
        width=600, height=700,
        plot_bgcolor="white",
    )
    fig.show()


def visualize_input_tensor_slices(tensor: np.ndarray, pose_rank: int):
    """
    Show the central XY, XZ, YZ slices of each input channel for one pose.
    Helps verify the interface box was extracted correctly.
    """
    mid = BOX_VOXELS // 2
    channel_names  = ["Shape", "Electrostatics", "Hydrophobicity"]
    slice_names    = ["XY (z=mid)", "XZ (y=mid)", "YZ (x=mid)"]
    cmaps          = ["RdBu_r", "RdBu_r", "YlOrRd"]

    fig = make_subplots(
        rows=N_CHANNELS, cols=3,
        subplot_titles=[f"{c}  —  {s}"
                        for c in channel_names for s in slice_names],
        shared_xaxes=False,
    )

    for row, (ch, cmap) in enumerate(zip(range(N_CHANNELS), cmaps), start=1):
        slices = [tensor[ch, :, :, mid],
                  tensor[ch, :, mid, :],
                  tensor[ch, mid, :, :]]
        for col, slc in enumerate(slices, start=1):
            vmax = max(abs(slc.min()), abs(slc.max())) + 1e-8
            fig.add_trace(
                go.Heatmap(z=slc, colorscale=cmap,
                           zmin=-vmax, zmax=vmax,
                           showscale=(col == 3),
                           colorbar=dict(y=1 - (row-1)/N_CHANNELS - 0.15,
                                         len=0.28)),
                row=row, col=col,
            )

    fig.update_layout(
        title=f"Input Tensor Slices  —  Pose rank {pose_rank}  (interface box {BOX_VOXELS}³ Å)",
        height=250 * N_CHANNELS,
        width=900,
    )
    fig.show()


def visualize_training_curves(train_losses: List[float], val_losses: List[float]):
    """Plot training and validation BCE loss curves."""
    epochs = list(range(1, len(train_losses) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode="lines+markers",
                             name="Training loss", line=dict(color="steelblue", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_losses, mode="lines+markers",
                             name="Validation loss", line=dict(color="tomato", width=2,
                                                               dash="dash")))
    fig.update_layout(
        title="3D-CNN Training — Binary Cross-Entropy Loss",
        xaxis_title="Epoch",
        yaxis_title="BCE Loss",
        width=800, height=450,
    )
    fig.show()


def visualize_score_reranking(poses: List[Pose]):
    """Scatter plot of FFT rank vs. CNN score, coloured by FFT score."""
    fft_ranks  = [p.fft_rank  for p in poses]
    cnn_scores = [p.cnn_score for p in poses]
    fft_scores = [p.fft_score for p in poses]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["FFT Rank vs. CNN Score", "CNN Score Distribution"],
    )

    fig.add_trace(
        go.Scatter(
            x=fft_ranks, y=cnn_scores,
            mode="markers",
            marker=dict(size=8, color=fft_scores, colorscale="Viridis",
                        colorbar=dict(title="FFT score", x=0.45)),
            text=[f"Rank {r}  FFT={s:.1f}  CNN={c:.3f}"
                  for r, s, c in zip(fft_ranks, fft_scores, cnn_scores)],
            name="Poses",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Histogram(x=cnn_scores, nbinsx=20,
                     marker_color="steelblue", opacity=0.7, name="CNN scores"),
        row=1, col=2,
    )

    # Highlight biggest rank change
    cnn_ranks = np.argsort(np.argsort([-s for s in cnn_scores])) + 1
    changes   = np.abs(np.array(fft_ranks) - cnn_ranks)
    top_idx   = int(np.argmax(changes))
    fig.add_annotation(
        x=fft_ranks[top_idx], y=cnn_scores[top_idx],
        text=f"  Largest re-rank<br>  FFT #{fft_ranks[top_idx]} → CNN #{cnn_ranks[top_idx]}",
        showarrow=True, arrowhead=2, row=1, col=1,
    )

    fig.update_xaxes(title_text="FFT rank",   row=1, col=1)
    fig.update_yaxes(title_text="CNN score",  row=1, col=1)
    fig.update_xaxes(title_text="CNN score",  row=1, col=2)
    fig.update_yaxes(title_text="Count",      row=1, col=2)
    fig.update_layout(title="Pose Re-ranking: FFT vs. 3D-CNN",
                      width=1000, height=480, showlegend=False)
    fig.show()


def visualize_gradcam(cam: np.ndarray, tensor: np.ndarray, pose_rank: int):
    """
    Overlay the Grad-CAM saliency map on the shape channel.
    High-saliency voxels (yellow) = regions most important to the CNN's decision.
    Low-saliency voxels = grey molecular surface for context.
    """
    shape_ch = tensor[0]   # (B, B, B)
    surf_idx = np.argwhere(shape_ch != 0)
    cam_idx  = np.argwhere(cam > 0.3)

    if len(surf_idx) > 30_000:
        surf_idx = surf_idx[np.random.default_rng(0).choice(len(surf_idx), 30_000, replace=False)]
    if len(cam_idx)  > 20_000:
        cam_idx  = cam_idx [np.random.default_rng(0).choice(len(cam_idx),  20_000, replace=False)]

    surf_vals = cam[surf_idx[:,0], surf_idx[:,1], surf_idx[:,2]]
    cam_vals  = cam[cam_idx[:,0],  cam_idx[:,1],  cam_idx[:,2]]

    fig = go.Figure()
    if len(surf_idx):
        fig.add_trace(go.Scatter3d(
            x=surf_idx[:,0], y=surf_idx[:,1], z=surf_idx[:,2],
            mode="markers",
            marker=dict(size=2, color=surf_vals, colorscale="Greys",
                        cmin=0, cmax=1, opacity=0.2),
            name="Interface structure",
        ))
    if len(cam_idx):
        fig.add_trace(go.Scatter3d(
            x=cam_idx[:,0], y=cam_idx[:,1], z=cam_idx[:,2],
            mode="markers",
            marker=dict(size=5, color=cam_vals, colorscale="YlOrRd",
                        cmin=0.3, cmax=1.0,
                        colorbar=dict(title="Saliency"),
                        opacity=0.8),
            name="Grad-CAM saliency  (cam > 0.3)",
        ))

    fig.update_layout(
        title=(f"Grad-CAM Saliency  (3D)  —  Pose rank {pose_rank}"
               f"  |  Yellow = drives score"),
        scene=dict(xaxis_title="X (vox)", yaxis_title="Y (vox)",
                   zaxis_title="Z (vox)"),
        width=900, height=750,
    )
    fig.show()


# ════════════════════════════════════════════════════════════════════════════
# Pose loader helpers
# ════════════════════════════════════════════════════════════════════════════

def load_poses_from_pkl(pkl_path: str, complex_id: str, top_n: int = 50) -> List[Pose]:
    """Load DockingResult objects from phase4 results.pkl and wrap as Pose."""
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    if complex_id not in results:
        raise KeyError(f"Complex {complex_id!r} not found in {pkl_path}. "
                       f"Available: {list(results.keys())}")

    raw = results[complex_id][:top_n]
    return [
        Pose(
            fft_rank=i+1,
            fft_score=float(r.score),
            rotation_matrix=r.rotation_matrix,
            translation_vector=r.translation_vector,
        )
        for i, r in enumerate(raw)
    ]


def generate_synthetic_poses(top_n: int = 30) -> List[Pose]:
    """
    Generate synthetic poses from SO(3) rotations with plausible FFT scores.
    Used for demo mode when no results.pkl is available.
    """
    rotations = generate_uniform_rotations(30.0)[:top_n]
    scores    = sorted(np.random.default_rng(42).uniform(50, 500, top_n),
                       reverse=True)
    return [
        Pose(
            fft_rank=i+1,
            fft_score=float(scores[i]),
            rotation_matrix=R,
            translation_vector=np.random.default_rng(i).uniform(-10, 10, 3),
        )
        for i, R in enumerate(rotations)
    ]


# ════════════════════════════════════════════════════════════════════════════
# Interactive entry point
# ════════════════════════════════════════════════════════════════════════════

def _prompt(msg: str, default: str = "") -> str:
    val = input(f"{msg} [{default}]: ").strip()
    return val if val else default


def main():
    _require_torch()

    SEP = "═" * 65
    print(f"\n{SEP}")
    print("  Advanced Visualization — Module 5: 3D-CNN AI Scoring")
    print(f"  Interface Box Extraction  |  CNN Training  |  Grad-CAM")
    print(f"{SEP}")
    print("  ⚠  This pipeline processes ONE complex at a time.")
    print()
    print("  Modes:")
    print("    A — Demo  (random weights, full pipeline preview)")
    print("    B — Self-supervised training on this complex  (~3–10 min CPU)")
    print("    C — Load saved checkpoint (.pt file)")
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

    mode = _prompt("  Mode (A / B / C)", "A").upper()
    if mode not in ("A", "B", "C"):
        mode = "A"

    pkl_path = ""
    if mode in ("A", "B"):
        pkl_path = _prompt(
            "  Path to results.pkl (leave blank to use synthetic poses)",
            "",
        )

    ckpt_path = ""
    if mode == "C":
        ckpt_path = _prompt("  Path to .pt checkpoint file", "cnn_scorer.pt")

    top_n_str = _prompt("  Number of top poses to score", "30")
    try:
        top_n = int(top_n_str)
    except ValueError:
        top_n = 30

    print(f"\n  Loading complex {complex_id} …")
    cases, _ = load_uu_cases(json_path, pdb_root)
    target = next((c for c in cases if c.complex_id == complex_id), None)

    if target is None:
        print(f"\n  ✗ Complex '{complex_id}' not found. Check ID and paths.")
        return
    print(f"  ✓ Loaded {complex_id}\n")

    # ── Load or generate poses ───────────────────────────────────────────────
    if pkl_path and os.path.isfile(pkl_path):
        print(f"  Loading poses from {pkl_path} …")
        try:
            poses = load_poses_from_pkl(pkl_path, complex_id, top_n)
            print(f"  ✓ Loaded {len(poses)} poses from pkl")
        except Exception as e:
            print(f"  ✗ Could not load pkl: {e} — falling back to synthetic poses")
            poses = generate_synthetic_poses(top_n)
    else:
        print("  Generating synthetic poses for demo …")
        poses = generate_synthetic_poses(top_n)
        print(f"  ✓ {len(poses)} synthetic poses generated")

    # ── Build interface tensors ──────────────────────────────────────────────
    extractor = InterfaceBoxExtractor()
    print(f"\n  Extracting {BOX_VOXELS}³ Å interface boxes  ({len(poses)} poses) …")
    print("  (This may take a few minutes — each box requires voxel grid construction.)")

    tensors = []
    for i, pose in enumerate(poses):
        if i % 10 == 0:
            print(f"    Pose {i+1}/{len(poses)} …")
        try:
            t = extractor.extract(pose, target.protein_struct, target.rna_struct)
        except Exception as e:
            print(f"    ⚠ Pose {i+1} extraction failed ({e}), substituting zeros")
            t = np.zeros((N_CHANNELS, BOX_VOXELS, BOX_VOXELS, BOX_VOXELS), dtype=np.float32)
        tensors.append(t)
    print(f"  ✓ All {len(tensors)} tensors extracted")

    # ── Build and optionally train model ────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device.type.upper()}")

    model = BindingScorer3DCNN()
    print(f"\n{model.architecture_str()}\n")

    train_losses, val_losses = [], []

    if mode == "B":
        # Positive = top third, Negative = bottom third
        n_pos = max(2, len(poses) // 3)
        n_neg = max(2, len(poses) // 3)
        pos_tensors = tensors[:n_pos]
        neg_tensors = tensors[-n_neg:]

        print(f"  Training: {n_pos} positives (top FFT), {n_neg} negatives (bottom FFT)")
        trainer = CNNTrainer(model, device, epochs=25)
        train_losses, val_losses = trainer.train(pos_tensors, neg_tensors)

        save_path = f"cnn_scorer_{complex_id}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"  ✓ Checkpoint saved to {save_path}")

    elif mode == "C":
        if os.path.isfile(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"  ✓ Weights loaded from {ckpt_path}")
        else:
            print(f"  ✗ Checkpoint not found at {ckpt_path!r} — using random weights")

    # ── Score all poses ──────────────────────────────────────────────────────
    model.eval()
    model.to(device)
    print("\n  Scoring all poses …")
    with torch.no_grad():
        for pose, tensor in zip(poses, tensors):
            x  = torch.tensor(tensor[None], dtype=torch.float32, device=device)
            pose.cnn_score = float(model(x).item())

    poses_by_cnn = sorted(poses, key=lambda p: p.cnn_score, reverse=True)
    print(f"\n  Top 5 poses by CNN score:")
    print(f"  {'CNN Rank':>8}  {'FFT Rank':>8}  {'CNN Score':>10}  {'FFT Score':>10}")
    print(f"  {'─'*44}")
    for cnn_rank, p in enumerate(poses_by_cnn[:5], 1):
        print(f"  {cnn_rank:>8}  {p.fft_rank:>8}  {p.cnn_score:>10.4f}  {p.fft_score:>10.1f}")

    # ── Grad-CAM for top CNN pose ────────────────────────────────────────────
    top_pose = poses_by_cnn[0]
    top_tensor = tensors[top_pose.fft_rank - 1]
    print(f"\n  Computing Grad-CAM for top CNN pose (FFT rank {top_pose.fft_rank}) …")
    cam = compute_gradcam_3d(model, top_tensor, device)
    print("  ✓ Grad-CAM computed")

    print(f"\n{SEP}")
    print("  Visualization Menu")
    print("  ─────────────────")
    print("  [1]  CNN architecture flowchart")
    print("  [2]  Input tensor channel slices  (top CNN pose)")
    if train_losses:
        print("  [3]  Training curves  (BCE loss)")
    print("  [4]  Pose re-ranking scatter  (FFT rank vs. CNN score)")
    print("  [5]  Grad-CAM saliency 3D  (top CNN pose)")
    print("  [A]  All of the above")
    print(f"{SEP}")

    choice = input("  Enter option(s) (e.g.  1 4 5  or  A): ").strip().upper()

    dispatch = {
        "1": lambda: visualize_architecture(model),
        "2": lambda: visualize_input_tensor_slices(top_tensor, top_pose.fft_rank),
        "3": (lambda: visualize_training_curves(train_losses, val_losses)) if train_losses
             else lambda: print("  No training data (Mode A / C used)"),
        "4": lambda: visualize_score_reranking(poses),
        "5": lambda: visualize_gradcam(cam, top_tensor, top_pose.fft_rank),
    }

    all_keys = list(dispatch.keys())
    keys  = all_keys if "A" in choice else choice.split()
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
