# Phase 1: Rigid-Body FFT-based Protein-RNA Docking

Welcome to Phase 1 of the **Prot-RNA-Docking-BTP** pipeline. This phase executes a global, rigid-body docking search to predict the 3D complex structure of a protein and an RNA molecule starting from their unbound (UU) states. 

The core philosophy of this phase relies on **shape complementarity**: finding the orientation and translation where the surface of the RNA perfectly nests into the surface of the protein, without their solid interiors clashing. Because the search space is massive (all possible 3D rotations and translations), we utilize a Fast Fourier Transform (FFT) approach combined with a uniform SO(3) rotation sampler to make the computation highly efficient.

---

## Pipeline Overview & Biological Rationale

Macromolecules interact primarily through non-covalent interactions (hydrogen bonding, electrostatic forces, and Van der Waals forces) across their accessible surfaces. However, two atoms cannot occupy the same physical space due to Pauli exclusion (steric clashing). 

This algorithm models these physical realities using the **Katchalski-Katzir representation** (1992). We map the continuous molecular structures into discrete 3D grids (voxels), assigning rewards for surface-to-surface contact and massive penalties for interior-to-interior overlaps.

### The Scripts in this Phase:
1. **`PDBparser.py`**: Extracts the atomic coordinates of the unbound protein and unbound RNA from standard PDB files.
2. **`grid3d.py`**: Discretizes the 3D coordinate space into a voxel grid, distinguishing between the molecular surface, interior, and empty space.
3. **`RotSampler.py`**: Generates a mathematically uniform set of 3D rotations to sample all possible orientations of the RNA molecule.
4. **`FFT.py`**: Evaluates all possible translational shifts of the rotated RNA against the static protein grid simultaneously using the Convolution Theorem.

---

## File-by-File Mathematical & Algorithmic Logic

### 1. `PDBparser.py` (Coordinate Extraction)
Proteins and RNA are represented as arrays of atoms with distinct 3D coordinates $(x, y, z)$. This script cleans the input, infers molecule types (Protein vs. RNA) based on standard residue dictionaries (e.g., ALA, ARG vs. ADE, URA), and filters out irrelevant atoms (like hydrogens, which are implicitly handled by heavy-atom Van der Waals radii in rigid-body docking).

### 2. `grid3d.py` (Voxelization)
To prepare for the FFT, the atomic coordinates are embedded into a unified 3D cartesian grid with dimensions $N_x \times N_y \times N_z$ (strictly powers of 2 for FFT efficiency).

Each voxel represents a small cubic volume of space (e.g., 1.0 Å³). The grid values are assigned based on distance to the atomic centers:
* **Interior (-15):** Voxels falling deeply inside the Van der Waals radii of the atoms. The large negative value heavily penalizes steric clashes.
* **Surface (+1):** Voxels on the outermost shell (typically 1.4 Å thick, representing the radius of a water probe). This represents the interaction boundary.
* **Open Space (0):** The solvent/vacuum.

### 3. `RotSampler.py` (SO(3) Rotation Space)
To find the right docking pose, we must test how the RNA fits against the protein from every possible angle. The space of all possible 3D rotations is the Special Orthogonal group, $SO(3)$. 

A naive approach (like uniformly sampling Euler angles) results in "gimbal lock" and oversampling at the poles. Instead, we map rotations to unit quaternions $q = (w, x, y, z)$ on the 3-sphere $S^3$. Using Hopf coordinates $(\theta, \phi, \psi)$, we generate a deterministic, uniformly distributed grid of rotations. 

For a given rotation matrix $R$, the RNA coordinates $\mathbf{x}$ are rotated around their geometric center $\mathbf{c}$:
$$\mathbf{x}' = R(\mathbf{x} - \mathbf{c}) + \mathbf{c}$$

### 4. `FFT.py` (Translational Search via Cross-Correlation)


Once the RNA is rotated, we need to find the optimal translation vector $\mathbf{t} = (t_x, t_y, t_z)$ that maximizes the shape complementarity score $S$. In the spatial domain, this is a cross-correlation between the Protein grid $P$ and the rotated RNA grid $R_{rot}$:
$$S(\mathbf{t}) = \sum_{\mathbf{r}} P(\mathbf{r}) \cdot R_{rot}(\mathbf{r} - \mathbf{t})$$

Calculating this directly for a $128 \times 128 \times 128$ grid requires billions of operations per rotation ($O(N^6)$). To bypass this, we use the **Convolution Theorem**, which states that spatial correlation is equivalent to point-wise multiplication in the frequency domain. 

We apply the Fast Fourier Transform ($\mathcal{F}$), multiply the transformed grids (taking the complex conjugate of the RNA grid), and apply the Inverse Fast Fourier Transform ($\mathcal{F}^{-1}$) to return to the spatial domain:
$$S = \mathcal{F}^{-1}(\mathcal{F}(P) \cdot \overline{\mathcal{F}(R_{rot})})$$

This reduces the complexity to $O(N^3 \log N)$. The voxel coordinate with the maximum peak in the resulting matrix $S$ represents the optimal translation vector for that specific rotation.

---

## How to Run Phase 1

1. Ensure your dataset (like PRDBv3) is correctly linked in your arguments.
2. Run the full search pipeline via the FFT master script:
```bash
python FFT.py --step 15.0 --resolution 1.0
```
3. The script will output the top scoring (Rotation, Translation) pairs, which will be passed to Phase 2 for structural refinement and energy minimization.
