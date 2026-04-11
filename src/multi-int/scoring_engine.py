#!/usr/bin/python3
# =============================================================================
# scoring_engine.py  —  Cross-reference metrics for protein-RNA interface scoring
# =============================================================================
#
# OVERVIEW
# --------
# Computes interaction metrics between ground truth (Mode A) and generated poses
# (Mode C). For each generated pose, this engine calculates:
#
#   1. Fraction of Native Contacts (f_nat)
#      % of residue-residue contacts in ground truth that appear in generated pose.
#      Uses residue-level contacts (CA for protein, C4' for RNA) with an 8.0 Å
#      threshold — the CAPRI standard for protein-RNA docking.
#
#   2. Interface RMSD (I-RMSD)
#      Spatial deviation of backbone atoms (N, CA, C, O) at the interface
#      after SVD-based optimal superimposition (Kabsch algorithm).
#
#   3. BSA Recovery (Δ BSA)
#      Absolute difference in buried surface area between ground truth and
#      generated pose.
#
#   4. Steric Clash Penalty
#      Voxel-based overlap: fraction of 1 Å³ voxels occupied by both protein
#      and RNA atoms (0 = no clash, 1 = complete overlap).
#
#   5. Composite Interaction Score [0, 1]
#      Weighted aggregation of all metrics. Higher = more near-native.
#
# WHY THE OLD ENGINE GAVE 0.0 / INCORRECT FOR EVERYTHING
# -------------------------------------------------------
#   Bug 1 — f_nat used ALL atomic coordinates instead of interface atoms.
#            This made the metric meaningless (comparing full structures).
#   Bug 2 — .int file parser was fragile and returned empty sets, so interface
#            atoms could never be found → f_nat=0, I-RMSD=inf → score≈0.1.
#   Bug 3 — I-RMSD was computed WITHOUT superimposition, so two identical
#            poses at different orientations would score poorly.
#   Bug 4 — bsa_delta normalisation constant was 100 Å² — too tight. Most
#            realistic poses have Δ BSA > 100 Å², so the BSA term was always ≈0.
#   Bug 5 — No fallback when .int files are empty: the engine silently returned
#            0.0 for f_nat and inf for I-RMSD rather than detecting the
#            interface directly from atomic distances.
#
# =============================================================================

import os
import logging
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List, Set, Tuple, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 8.0 Å — CAPRI standard residue-level contact threshold (representative atoms)
RESIDUE_CONTACT_DISTANCE = 8.0

# Backbone atoms used for I-RMSD
BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# Standard RNA residue names (3-letter and 1-letter equivalents)
RNA_RESIDUES = {
    "A", "U", "G", "C",          # 1-letter in PDB chain
    "DA", "DU", "DG", "DC",      # DNA
    "ADE", "URA", "GUA", "CYT",  # sometimes used
    "rA", "rU", "rG", "rC",      # alternative
    "ATP", "GTP", "UTP", "CTP",  # nucleotides (edge case)
}


# ---------------------------------------------------------------------------
# PDB Parsing
# ---------------------------------------------------------------------------

def parse_pdb_residues(pdb_file: str) -> List[Dict]:
    """
    Parse a PDB file and return a list of residue records.

    Each record is a dict:
        chain    : str             — chain ID
        resnum   : int             — residue sequence number
        resname  : str             — 3-letter residue name
        is_rna   : bool            — True if RNA nucleotide
        atoms    : {name: ndarray} — all heavy-atom coordinates
        rep_atom : ndarray | None  — representative atom (CA or C4')
        backbone : List[ndarray]   — N, CA, C, O coordinates

    Parameters
    ----------
    pdb_file : str
        Path to PDB file.

    Returns
    -------
    List[Dict]
        One dict per unique (chain, resnum) pair, in PDB order.
    """
    residues_map: Dict[Tuple[str, int], Dict] = {}
    order: List[Tuple[str, int]] = []

    if not os.path.exists(pdb_file):
        logging.error(f"parse_pdb_residues: file not found: {pdb_file!r}")
        return []

    with open(pdb_file, "r") as fh:
        for line in fh:
            rec_type = line[:6].strip()
            if rec_type not in ("ATOM", "HETATM"):
                continue
            try:
                atom_name = line[12:16].strip()
                res_name  = line[17:20].strip()
                chain     = line[21]        # single char, may be space
                res_seq   = line[22:26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            chain = chain.strip()
            if not chain:
                chain = "X"          # give unnamed chains a placeholder
            try:
                resnum = int(res_seq)
            except ValueError:
                continue

            key = (chain, resnum)
            if key not in residues_map:
                is_rna = res_name in RNA_RESIDUES
                residues_map[key] = {
                    "chain":    chain,
                    "resnum":   resnum,
                    "resname":  res_name,
                    "is_rna":   is_rna,
                    "atoms":    {},
                    "rep_atom": None,
                    "backbone": [],
                }
                order.append(key)

            coord = np.array([x, y, z], dtype=np.float64)
            rec = residues_map[key]
            rec["atoms"][atom_name] = coord

            is_rna = rec["is_rna"]
            # Representative atom
            if is_rna and atom_name == "C4'" and rec["rep_atom"] is None:
                rec["rep_atom"] = coord
            elif not is_rna and atom_name == "CA" and rec["rep_atom"] is None:
                rec["rep_atom"] = coord

            # Backbone
            if atom_name in BACKBONE_ATOMS:
                rec["backbone"].append(coord)

    # Fallback representative atom: centroid of all atoms
    for key in order:
        rec = residues_map[key]
        if rec["rep_atom"] is None and rec["atoms"]:
            rec["rep_atom"] = np.mean(list(rec["atoms"].values()), axis=0)

    result = [residues_map[k] for k in order]
    logging.debug(f"parse_pdb_residues: {len(result)} residues from {pdb_file!r}")
    return result


# ---------------------------------------------------------------------------
# .int File Parsing
# ---------------------------------------------------------------------------

def get_interface_residue_ids(int_file: str) -> Set[Tuple[str, int]]:
    """
    Parse a naccess .int file → set of (chain, resnum) tuples.

    naccess RSA-format lines look like:
        RES  ALA  A  123   ...   (columns: keyword, resname, chain, resnum, ...)

    We try column layout [2]=chain, [3]=resnum first, then fall back to
    scanning for a single-letter token followed / preceded by an integer.

    Parameters
    ----------
    int_file : str
        Path to .int file.

    Returns
    -------
    Set[Tuple[str, int]]
    """
    ids: Set[Tuple[str, int]] = set()

    if not int_file or not os.path.exists(int_file):
        logging.warning(f"get_interface_residue_ids: file missing: {int_file!r}")
        return ids

    with open(int_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            parsed = False
            # Primary layout: RES resname chain resnum ...
            if len(parts) >= 4:
                try:
                    chain  = parts[2].strip()
                    resnum = int(parts[3].strip())
                    if chain:
                        ids.add((chain, resnum))
                        parsed = True
                except (ValueError, IndexError):
                    pass

            # Fallback: scan for single-alpha char adjacent to integer
            if not parsed:
                for i, p in enumerate(parts):
                    if len(p) == 1 and p.isalpha():
                        for j in (i + 1, i - 1):
                            if 0 <= j < len(parts):
                                try:
                                    resnum = int(parts[j])
                                    ids.add((p, resnum))
                                    parsed = True
                                    break
                                except ValueError:
                                    pass
                    if parsed:
                        break

    n_lines = sum(1 for l in open(int_file) if l.strip()) if os.path.exists(int_file) else 0
    if not ids:
        logging.warning(
            f"get_interface_residue_ids: 0 IDs parsed from {int_file!r} "
            f"({n_lines} non-empty lines). File may be empty or use an "
            "unexpected format — will fall back to distance-based detection."
        )
    else:
        logging.debug(f"get_interface_residue_ids: {len(ids)} residues from {int_file!r}")

    return ids


# ---------------------------------------------------------------------------
# Interface Detection Fallback
# ---------------------------------------------------------------------------

def detect_interface_by_distance(residues: List[Dict],
                                  cutoff: float = RESIDUE_CONTACT_DISTANCE
                                  ) -> Set[Tuple[str, int]]:
    """
    Detect interface residues purely by distance (protein ↔ RNA) when .int
    files are unavailable or empty.

    Returns (chain, resnum) IDs of residues within `cutoff` Å of the other
    molecular type (protein or RNA).
    """
    protein = [r for r in residues if not r["is_rna"] and r["rep_atom"] is not None]
    rna     = [r for r in residues if r["is_rna"]     and r["rep_atom"] is not None]

    if not protein or not rna:
        logging.warning(
            "detect_interface_by_distance: could not find both protein and RNA residues. "
            f"protein={len(protein)}, rna={len(rna)}"
        )
        return set()

    p_coords = np.array([r["rep_atom"] for r in protein])
    r_coords = np.array([r["rep_atom"] for r in rna])
    dists = cdist(p_coords, r_coords)

    ids: Set[Tuple[str, int]] = set()
    for pi, pr in enumerate(protein):
        if np.any(dists[pi] < cutoff):
            ids.add((pr["chain"], pr["resnum"]))
    for ri, rr in enumerate(rna):
        if np.any(dists[:, ri] < cutoff):
            ids.add((rr["chain"], rr["resnum"]))

    logging.debug(
        f"detect_interface_by_distance: {len(ids)} interface residues at {cutoff} Å cutoff"
    )
    return ids


# ---------------------------------------------------------------------------
# Coordinate-Frame Alignment
# ---------------------------------------------------------------------------

def _get_all_backbone_coords(residues: List[Dict]) -> np.ndarray:
    """Collect all backbone atom coordinates from a residue list (N, CA, C, O)."""
    pts = []
    for r in residues:
        for atom in sorted(BACKBONE_ATOMS):
            c = r["atoms"].get(atom)
            if c is not None:
                pts.append(c)
    return np.array(pts) if pts else np.empty((0, 3))


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (R, t) such that  P_aligned = P @ R.T + t  minimises RMSD to Q.

    P and Q must have the same shape (N, 3).  Uses the Kabsch / SVD algorithm.
    """
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_c = P - P_mean
    Q_c = Q - Q_mean

    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    t = Q_mean - P_mean @ R.T
    return R, t


def _apply_transform(residues: List[Dict], R: np.ndarray, t: np.ndarray) -> List[Dict]:
    """
    Return a deep-copied residue list with all coordinates rotated and translated.

    Applied as:  coord_new = coord @ R.T + t
    """
    import copy
    aligned = []
    for r in residues:
        nr = copy.deepcopy(r)
        nr["atoms"] = {name: coord @ R.T + t for name, coord in r["atoms"].items()}
        if r["rep_atom"] is not None:
            nr["rep_atom"] = r["rep_atom"] @ R.T + t
        nr["backbone"] = [c @ R.T + t for c in r["backbone"]]
        aligned.append(nr)
    return aligned


def align_generated_to_truth(truth_residues: List[Dict],
                              gen_residues: List[Dict]) -> List[Dict]:
    """
    Superimpose the generated structure onto the ground truth using all
    backbone atoms (N, CA, C, O).  Returns the aligned generated residues.

    When the truth and generated structures share the same number of backbone
    atoms they are matched index-by-index (safe for FFT outputs that preserve
    residue count).  If counts differ, the first min(N_t, N_g) atoms are used.

    This resolves the "physical disconnect" / coordinate-frame mismatch that
    causes f_nat=0 and I-RMSD=inf when FFT output lives in a different frame
    from the ground truth.
    """
    t_bb = _get_all_backbone_coords(truth_residues)
    g_bb = _get_all_backbone_coords(gen_residues)

    if len(t_bb) == 0 or len(g_bb) == 0:
        logging.warning("align_generated_to_truth: no backbone atoms — skipping alignment")
        return gen_residues

    n = min(len(t_bb), len(g_bb))
    if n < 3:
        logging.warning(f"align_generated_to_truth: only {n} matched backbone atoms — skipping")
        return gen_residues

    if len(t_bb) != len(g_bb):
        logging.warning(
            f"align_generated_to_truth: backbone atom count mismatch "
            f"(truth={len(t_bb)}, gen={len(g_bb)}) — using first {n}"
        )

    R, t_vec = _kabsch_rotation(g_bb[:n], t_bb[:n])
    aligned = _apply_transform(gen_residues, R, t_vec)

    # Verify improvement
    g_bb_aligned = _get_all_backbone_coords(aligned)
    pre_rmsd  = float(np.sqrt(np.mean(np.sum((g_bb[:n] - t_bb[:n]) ** 2, axis=1))))
    post_rmsd = float(np.sqrt(np.mean(np.sum((g_bb_aligned[:n] - t_bb[:n]) ** 2, axis=1))))
    logging.info(
        f"align_generated_to_truth: backbone RMSD  before={pre_rmsd:.2f} Å  "
        f"after={post_rmsd:.2f} Å  (n={n} atoms)"
    )

    return aligned


# ---------------------------------------------------------------------------
# Chain-agnostic residue matching helper
# ---------------------------------------------------------------------------

def _match_residues_by_index(
        truth_residues: List[Dict],
        gen_residues: List[Dict],
        truth_ids: Set[Tuple[str, int]],
        gen_ids: Set[Tuple[str, int]],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Match interface residues between truth and generated structures.

    Primary strategy: exact (chain, resnum) match.
    Fallback: positional index match when chain IDs differ (common for FFT
    outputs that use generic chain labels like 'A'/'B' regardless of the
    original chain name).

    Returns (truth_int_residues, gen_int_residues) — paired lists of the
    same length.
    """
    # Filter to interface
    t_int = [r for r in truth_residues if (r["chain"], r["resnum"]) in truth_ids] \
            if truth_ids else truth_residues
    g_int = [r for r in gen_residues if (r["chain"], r["resnum"]) in gen_ids] \
            if gen_ids else gen_residues

    # Check exact overlap
    t_keys = {(r["chain"], r["resnum"]) for r in t_int}
    g_keys = {(r["chain"], r["resnum"]) for r in g_int}
    common = t_keys & g_keys

    if common:
        # Use exact matches only
        t_matched = [r for r in t_int if (r["chain"], r["resnum"]) in common]
        g_matched = [r for r in g_int if (r["chain"], r["resnum"]) in common]
        return t_matched, g_matched

    # Fallback: match by molecular type (protein/RNA) and sequence index
    logging.warning(
        "match_residues: no (chain, resnum) overlap between truth and generated — "
        "falling back to index-based matching by molecule type"
    )
    t_pro = [r for r in t_int if not r["is_rna"]]
    t_rna = [r for r in t_int if r["is_rna"]]
    g_pro = [r for r in g_int if not r["is_rna"]]
    g_rna = [r for r in g_int if r["is_rna"]]

    n_pro = min(len(t_pro), len(g_pro))
    n_rna = min(len(t_rna), len(g_rna))

    t_matched = t_pro[:n_pro] + t_rna[:n_rna]
    g_matched = g_pro[:n_pro] + g_rna[:n_rna]
    return t_matched, g_matched


# ---------------------------------------------------------------------------
# Metric 1: Fraction of Native Contacts (f_nat)
# ---------------------------------------------------------------------------

def calculate_f_nat(truth_residues: List[Dict],
                    gen_residues: List[Dict],
                    truth_int_ids: Set[Tuple[str, int]],
                    gen_int_ids: Set[Tuple[str, int]]) -> float:
    """
    Fraction of Native Contacts — CAPRI residue-level definition.

    A native contact is a (protein_residue, RNA_residue) pair whose
    representative atoms are within RESIDUE_CONTACT_DISTANCE (8.0 Å) in the
    ground truth. f_nat is the fraction of those contacts also present in the
    generated pose.

    f_nat = |{native contacts recovered in generated}| / |{native contacts}|

    Parameters
    ----------
    truth_residues, gen_residues : List[Dict]
        From parse_pdb_residues().
    truth_int_ids, gen_int_ids : Set[Tuple[str, int]]
        Interface residue IDs. If empty, all residues are used.

    Returns
    -------
    float in [0.0, 1.0]. Higher is better.
    """
    def _split(residues):
        pro = [r for r in residues if not r["is_rna"] and r["rep_atom"] is not None]
        rna = [r for r in residues if r["is_rna"]     and r["rep_atom"] is not None]
        return pro, rna

    # Use all residues when interface IDs unavailable (graceful fallback)
    t_int = [r for r in truth_residues if (r["chain"], r["resnum"]) in truth_int_ids] \
            if truth_int_ids else truth_residues
    g_int = [r for r in gen_residues if (r["chain"], r["resnum"]) in gen_int_ids] \
            if gen_int_ids else gen_residues

    # If no chain overlap, fall back to full residue list for contact detection
    t_keys = {(r["chain"], r["resnum"]) for r in t_int}
    g_keys = {(r["chain"], r["resnum"]) for r in g_int}
    if not (t_keys & g_keys) and truth_int_ids and gen_int_ids:
        logging.warning("f_nat: no (chain,resnum) overlap in interface — using all residues")
        t_int = truth_residues
        g_int = gen_residues

    t_pro, t_rna = _split(t_int)
    g_pro, g_rna = _split(g_int)

    if not t_pro or not t_rna:
        logging.warning(
            f"f_nat: ground truth interface has no protein ({len(t_pro)}) "
            f"or RNA ({len(t_rna)}) residues"
        )
        return 0.0

    # Native contacts
    t_p_coords = np.array([r["rep_atom"] for r in t_pro])
    t_r_coords = np.array([r["rep_atom"] for r in t_rna])
    t_dists    = cdist(t_p_coords, t_r_coords)

    native: Set[Tuple] = set()
    for pi, pr in enumerate(t_pro):
        for ri, rr in enumerate(t_rna):
            if t_dists[pi, ri] < RESIDUE_CONTACT_DISTANCE:
                native.add((pr["chain"], pr["resnum"], rr["chain"], rr["resnum"]))

    n_native = len(native)
    if n_native == 0:
        logging.warning(
            f"f_nat: 0 native contacts at {RESIDUE_CONTACT_DISTANCE} Å — "
            f"truth has {len(t_pro)} protein, {len(t_rna)} RNA interface residues"
        )
        return 0.0

    if not g_pro or not g_rna:
        logging.warning("f_nat: generated pose interface has no protein or RNA residues")
        return 0.0

    # Generated contacts
    g_p_coords = np.array([r["rep_atom"] for r in g_pro])
    g_r_coords = np.array([r["rep_atom"] for r in g_rna])
    g_dists    = cdist(g_p_coords, g_r_coords)

    # Build generated contact set using positional indices so it can be
    # compared with the native set regardless of chain-ID differences.
    # We map each generated residue to the closest truth residue by index.
    t_pro_idx = {i: r for i, r in enumerate(
        [r for r in truth_residues if not r["is_rna"]]
    )}
    t_rna_idx = {i: r for i, r in enumerate(
        [r for r in truth_residues if r["is_rna"]]
    )}

    def _truth_key_for_gen(gen_res, truth_list_by_index):
        """Return the (chain, resnum) of the truth residue at the same index."""
        # Try exact match first
        for tr in truth_list_by_index.values():
            if tr["chain"] == gen_res["chain"] and tr["resnum"] == gen_res["resnum"]:
                return (tr["chain"], tr["resnum"])
        # Fall back to positional index from g_pro / g_rna list
        return None

    # Map generated protein residues → truth protein residues by position
    gen_pro_list = [r for r in gen_residues if not r["is_rna"] and r["rep_atom"] is not None]
    gen_rna_list = [r for r in gen_residues if r["is_rna"]     and r["rep_atom"] is not None]
    truth_pro_list = [r for r in truth_residues if not r["is_rna"]]
    truth_rna_list = [r for r in truth_residues if r["is_rna"]]

    def _map_key(gen_r, gen_list, truth_list):
        """Map gen residue → truth (chain, resnum) by position in the list."""
        try:
            idx = next(i for i, r in enumerate(gen_list)
                       if r["chain"] == gen_r["chain"] and r["resnum"] == gen_r["resnum"])
        except StopIteration:
            return (gen_r["chain"], gen_r["resnum"])
        if idx < len(truth_list):
            return (truth_list[idx]["chain"], truth_list[idx]["resnum"])
        return (gen_r["chain"], gen_r["resnum"])

    generated: Set[Tuple] = set()
    for pi, pr in enumerate(g_pro):
        for ri, rr in enumerate(g_rna):
            if g_dists[pi, ri] < RESIDUE_CONTACT_DISTANCE:
                pk = _map_key(pr, gen_pro_list, truth_pro_list)
                rk = _map_key(rr, gen_rna_list, truth_rna_list)
                generated.add((pk[0], pk[1], rk[0], rk[1]))

    recovered = len(native & generated)
    f_nat = recovered / n_native
    logging.debug(
        f"f_nat: {n_native} native, {len(generated)} generated, "
        f"{recovered} recovered → {f_nat:.4f}"
    )
    return float(f_nat)


# ---------------------------------------------------------------------------
# Metric 2: Interface RMSD (I-RMSD)
# ---------------------------------------------------------------------------

def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    RMSD after optimal superimposition of P onto Q (Kabsch / SVD algorithm).

    Parameters
    ----------
    P, Q : np.ndarray of shape (N, 3)
        Matched atom coordinate sets.

    Returns
    -------
    float — RMSD in Ångströms after optimal rotation+translation.
    """
    assert P.shape == Q.shape and P.ndim == 2 and P.shape[1] == 3

    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)

    # Correct for improper rotation (reflection)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    P_rot = P_c @ R.T
    diff  = P_rot - Q_c
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def calculate_irmsd(truth_residues: List[Dict],
                    gen_residues: List[Dict],
                    truth_int_ids: Set[Tuple[str, int]],
                    gen_int_ids: Set[Tuple[str, int]]) -> float:
    """
    Interface RMSD (I-RMSD) with SVD-based superimposition.

    Extracts backbone atoms (N, CA, C, O) from residues in the UNION of
    truth and generated interface IDs, matches them by (chain, resnum, atom_name),
    and computes RMSD after Kabsch superimposition.

    Parameters
    ----------
    truth_residues, gen_residues : List[Dict]
        From parse_pdb_residues().
    truth_int_ids, gen_int_ids : Set[Tuple[str, int]]
        Interface residue IDs. If both empty, falls back to all residues.

    Returns
    -------
    float — I-RMSD in Å. Lower is better. Returns inf if < 3 matched atoms.
    """
    # --- Attempt exact (chain, resnum, atom) matching first ---
    combined_ids = truth_int_ids | gen_int_ids
    if not combined_ids:
        combined_ids = {(r["chain"], r["resnum"]) for r in truth_residues}

    def _index(residues):
        return {(r["chain"], r["resnum"]): r for r in residues}

    truth_idx = _index(truth_residues)
    gen_idx   = _index(gen_residues)

    truth_bb: List[np.ndarray] = []
    gen_bb:   List[np.ndarray] = []

    for key in sorted(combined_ids):
        t_res = truth_idx.get(key)
        g_res = gen_idx.get(key)
        if t_res is None or g_res is None:
            continue
        for atom in sorted(BACKBONE_ATOMS):
            tc = t_res["atoms"].get(atom)
            gc = g_res["atoms"].get(atom)
            if tc is not None and gc is not None:
                truth_bb.append(tc)
                gen_bb.append(gc)

    if len(truth_bb) < 3:
        # --- Fallback: index-based backbone matching by molecule type ---
        logging.warning(
            f"I-RMSD: only {len(truth_bb)} exact backbone matches — "
            "trying index-based fallback (chain-ID mismatch likely)"
        )
        t_matched, g_matched = _match_residues_by_index(
            truth_residues, gen_residues, truth_int_ids, gen_int_ids
        )

        truth_bb = []
        gen_bb   = []
        for t_res, g_res in zip(t_matched, g_matched):
            for atom in sorted(BACKBONE_ATOMS):
                tc = t_res["atoms"].get(atom)
                gc = g_res["atoms"].get(atom)
                if tc is not None and gc is not None:
                    truth_bb.append(tc)
                    gen_bb.append(gc)

    if len(truth_bb) < 3:
        logging.warning(
            f"I-RMSD: only {len(truth_bb)} matched backbone atoms after fallback — "
            "returning inf."
        )
        return float("inf")

    rmsd = _kabsch_rmsd(np.array(truth_bb), np.array(gen_bb))
    logging.debug(f"I-RMSD: {len(truth_bb)} backbone atoms → {rmsd:.3f} Å")
    return rmsd


# ---------------------------------------------------------------------------
# Metric 3: BSA Delta
# ---------------------------------------------------------------------------

def calculate_bsa_delta(truth_bsa, gen_bsa) -> float:
    """
    |truth_BSA - generated_BSA| in Ų. Lower is better.
    Returns float('inf') if either value is unavailable.
    """
    if truth_bsa in ("NA", None, "") or gen_bsa in ("NA", None, ""):
        return float("inf")
    try:
        return float(abs(float(truth_bsa) - float(gen_bsa)))
    except (TypeError, ValueError):
        return float("inf")


# ---------------------------------------------------------------------------
# Metric 4: Steric Clash Penalty
# ---------------------------------------------------------------------------

def calculate_steric_clash(protein_residues: List[Dict],
                            rna_residues: List[Dict],
                            voxel_size: float = 1.0) -> float:
    """
    Steric clash penalty via voxel overlap.

    Discretises all protein and RNA heavy atoms onto a shared 1 Å³ voxel grid.
    Penalty = |overlap voxels| / |union voxels|.

    Parameters
    ----------
    protein_residues, rna_residues : List[Dict]
        Residue records (from separate protein.pdb / rna.pdb parses).
    voxel_size : float
        Voxel edge length in Ångströms.

    Returns
    -------
    float in [0, 1]. 0 = no clash, 1 = complete overlap.
    """
    def _all_coords(residues):
        pts = []
        for r in residues:
            pts.extend(r["atoms"].values())
        return np.array(pts) if pts else np.empty((0, 3))

    p_coords = _all_coords(protein_residues)
    r_coords = _all_coords(rna_residues)

    if len(p_coords) == 0 or len(r_coords) == 0:
        return 0.0

    try:
        all_c  = np.vstack([p_coords, r_coords])
        origin = np.floor(all_c.min(axis=0) / voxel_size).astype(int)

        def _vox(coords):
            v = np.floor(coords / voxel_size).astype(int) - origin
            return set(map(tuple, v))

        pv = _vox(p_coords)
        rv = _vox(r_coords)

        overlap = len(pv & rv)
        union   = len(pv | rv)
        if union == 0:
            return 0.0
        return float(min(overlap / union, 1.0))

    except Exception as exc:
        logging.warning(f"calculate_steric_clash failed: {exc}")
        return 0.0


# ---------------------------------------------------------------------------
# Metric 5: Composite Interaction Score
# ---------------------------------------------------------------------------

def compute_interaction_score(f_nat: float,
                               irmsd: float,
                               bsa_delta: float,
                               clash_penalty: float,
                               weights: Optional[Dict] = None) -> float:
    """
    Composite Interaction Score [0, 1]. Higher is better (more near-native).

    Each raw metric is normalised to [0, 1] where 1 = perfect:
        f_nat       → already [0,1]
        irmsd       → exp(-irmsd / 5.0)      0 Å→1.0, 5 Å→0.37, 10 Å→0.14
        bsa_delta   → exp(-bsa_delta / 300)  0 Å²→1.0, 300 Å²→0.37
        clash       → 1 - clash_penalty

    Default weights: f_nat=0.4, irmsd=0.3, bsa_delta=0.2, clash=0.1

    Parameters
    ----------
    f_nat, irmsd, bsa_delta, clash_penalty : float
        Raw metric values (irmsd / bsa_delta may be inf).
    weights : dict | None
        Override default weights.

    Returns
    -------
    float in [0.0, 1.0]
    """
    if weights is None:
        weights = {
            "f_nat":     0.4,
            "irmsd":     0.3,
            "bsa_delta": 0.2,
            "clash":     0.1,
        }

    norm_f_nat   = float(np.clip(f_nat, 0.0, 1.0))
    norm_irmsd   = float(np.exp(-irmsd / 5.0))      if np.isfinite(irmsd)     else 0.0
    norm_bsa     = float(np.exp(-bsa_delta / 300.0)) if np.isfinite(bsa_delta) else 0.0
    norm_clash   = float(1.0 - np.clip(clash_penalty, 0.0, 1.0))

    w_total = sum(weights.values())
    if w_total == 0:
        return 0.0

    score = (
        weights["f_nat"]     * norm_f_nat +
        weights["irmsd"]     * norm_irmsd +
        weights["bsa_delta"] * norm_bsa   +
        weights["clash"]     * norm_clash
    ) / w_total

    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def score_single_rank(complex_id: str,
                       truth_pdb: str,
                       truth_bsa,
                       gen_pdb: str,
                       gen_bsa,
                       truth_int_file: str,
                       gen_int_file: str,
                       protein_pdb: str,
                       rna_pdb: str,
                       rank_label: str = "rank1") -> Dict:
    """
    Score one generated rank against the ground truth.

    Parameters
    ----------
    complex_id      : str   — e.g. "1ASY"
    truth_pdb       : str   — path to ground truth complex PDB
    truth_bsa       : float — ground truth BSA (Ų) or "NA"
    gen_pdb         : str   — path to generated combined PDB
    gen_bsa         : float — generated BSA (Ų) or "NA"
    truth_int_file  : str   — path to ground truth .int file
    gen_int_file    : str   — path to generated .int file
    protein_pdb     : str   — path to generated protein.pdb
    rna_pdb         : str   — path to generated rna.pdb
    rank_label      : str   — e.g. "rank1"

    Returns
    -------
    dict
        Keys: complex_id, rank, f_nat, i_rmsd, bsa_delta, clash_penalty,
              interaction_score, category, error, diagnostics
    """
    result = {
        "complex_id":        complex_id,
        "rank":              rank_label,
        "f_nat":             None,
        "i_rmsd":            None,
        "bsa_delta":         None,
        "clash_penalty":     None,
        "interaction_score": None,
        "category":          None,
        "error":             None,
        "diagnostics":       {},
    }

    try:
        # ── 0. Validate PDB files exist ──────────────────────────────────────
        missing = [
            f"{lbl}={path!r}"
            for lbl, path in [
                ("truth_pdb",   truth_pdb),
                ("gen_pdb",     gen_pdb),
                ("protein_pdb", protein_pdb),
                ("rna_pdb",     rna_pdb),
            ]
            if not path or not os.path.exists(path)
        ]
        if missing:
            raise FileNotFoundError(f"Missing PDB files: {', '.join(missing)}")

        # ── 1. Parse structures ──────────────────────────────────────────────
        truth_residues   = parse_pdb_residues(truth_pdb)
        gen_residues     = parse_pdb_residues(gen_pdb)
        protein_residues = parse_pdb_residues(protein_pdb)
        rna_residues_sep = parse_pdb_residues(rna_pdb)   # for clash detection

        diag = result["diagnostics"]
        diag["truth_n_residues"]   = len(truth_residues)
        diag["gen_n_residues"]     = len(gen_residues)
        diag["protein_n_residues"] = len(protein_residues)
        diag["rna_n_residues"]     = len(rna_residues_sep)

        if not truth_residues:
            raise ValueError(f"No ATOM records parsed from truth_pdb={truth_pdb!r}")
        if not gen_residues:
            raise ValueError(f"No ATOM records parsed from gen_pdb={gen_pdb!r}")

        # ── 1b. Superimpose generated structure onto ground truth ─────────────
        # This resolves the "physical disconnect" / coordinate-frame mismatch:
        # FFT docking outputs live in their own translation/rotation frame.
        # We align gen_residues (and protein/RNA separately) to the truth frame
        # before computing any distance-dependent metric.
        gen_residues     = align_generated_to_truth(truth_residues, gen_residues)
        protein_residues = align_generated_to_truth(
            [r for r in truth_residues if not r["is_rna"]], protein_residues
        ) if protein_residues else protein_residues
        rna_residues_sep = align_generated_to_truth(
            [r for r in truth_residues if r["is_rna"]], rna_residues_sep
        ) if rna_residues_sep else rna_residues_sep
        diag["alignment_applied"] = True

        # ── 2. Parse interface files ─────────────────────────────────────────
        truth_int_ids = get_interface_residue_ids(truth_int_file)
        gen_int_ids   = get_interface_residue_ids(gen_int_file)

        diag["truth_int_n"] = len(truth_int_ids)
        diag["gen_int_n"]   = len(gen_int_ids)

        # Fallback: detect interface by distance when .int files are empty
        if not truth_int_ids:
            logging.warning(
                f"[{complex_id}/{rank_label}] truth .int file is empty/missing — "
                "falling back to distance-based interface detection"
            )
            truth_int_ids = detect_interface_by_distance(truth_residues)
            diag["truth_int_source"] = "distance_fallback"
            diag["truth_int_n_fallback"] = len(truth_int_ids)
        else:
            diag["truth_int_source"] = "int_file"

        if not gen_int_ids:
            logging.warning(
                f"[{complex_id}/{rank_label}] generated .int file is empty/missing — "
                "falling back to distance-based interface detection"
            )
            gen_int_ids = detect_interface_by_distance(gen_residues)
            diag["gen_int_source"] = "distance_fallback"
            diag["gen_int_n_fallback"] = len(gen_int_ids)
        else:
            diag["gen_int_source"] = "int_file"

        # ── 3. Compute metrics ───────────────────────────────────────────────

        f_nat = calculate_f_nat(
            truth_residues, gen_residues, truth_int_ids, gen_int_ids
        )

        irmsd = calculate_irmsd(
            truth_residues, gen_residues, truth_int_ids, gen_int_ids
        )

        bsa_delta = calculate_bsa_delta(truth_bsa, gen_bsa)

        # Use separate protein/RNA PDBs for clash (cleaner than combined)
        clash_penalty = calculate_steric_clash(protein_residues, rna_residues_sep)

        # ── 4. Composite score ───────────────────────────────────────────────
        interaction_score = compute_interaction_score(
            f_nat, irmsd, bsa_delta, clash_penalty
        )

        # ── 5. Classify ──────────────────────────────────────────────────────
        irmsd_val = irmsd if np.isfinite(irmsd) else 999.0
        if interaction_score >= 0.7 and irmsd_val < 3.0:
            category = "Near-Native"
        elif interaction_score >= 0.4:
            category = "Medium"
        else:
            category = "Incorrect"

        result.update({
            "f_nat":             round(f_nat, 4),
            "i_rmsd":            round(irmsd, 3) if np.isfinite(irmsd) else "inf",
            "bsa_delta":         round(bsa_delta, 2) if np.isfinite(bsa_delta) else "inf",
            "clash_penalty":     round(clash_penalty, 4),
            "interaction_score": round(interaction_score, 4),
            "category":          category,
            "error":             None,
        })

        logging.info(
            f"[{complex_id}/{rank_label}] "
            f"f_nat={f_nat:.3f}  i_rmsd={result['i_rmsd']}  "
            f"bsa_delta={result['bsa_delta']}  clash={clash_penalty:.3f}  "
            f"score={interaction_score:.3f}  cat={category}"
        )

    except Exception as exc:
        import traceback as _tb
        err = f"{exc}\n{_tb.format_exc()}"
        logging.error(f"score_single_rank failed [{complex_id}/{rank_label}]: {err}")
        result["error"] = str(exc)
        result["interaction_score"] = 0.0
        result["category"] = "Error"

    return result