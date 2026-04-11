#!/usr/bin/python3
# =============================================================================
# multi_interface.py  —  Protein–RNA interface analysis engine
# =============================================================================
#
# OVERVIEW
# --------
# Computes per-residue buried surface area (BSA) and interface .int files for
# a protein chain and an RNA chain using naccess.  Supports three run modes:
#
#   "complex"   (default / original behaviour)
#               Extract protein and RNA chains from a single combined PDB file
#               using fetch_atomline().  Used for source-truth (bound) complexes
#               read from ALL_PDBs/<ID>/<ID>.pdb.
#
#   "unbound"   Extract each chain from its own separate PDB file.
#               The protein comes from U_pro_PDB (e.g. 1EOV.pdb) and the RNA
#               comes from U_RNA_PDB (e.g. 2TRA.pdb).  Both files live in the
#               same folder as the complex PDB (ALL_PDBs/<C_PDB>/).
#               Chain IDs are taken from U_PRO_chain / U_RNA_chain in the JSON.
#               A synthetic combined PDB is assembled from both for naccess.
#
#   "generated" Skip chain extraction entirely.  protein.pdb and rna.pdb are
#               already-split files produced by the FFT docking pipeline and
#               live in generated_PDBS/<ID>/rank<N>/.
#               All output files are written to a "results/" sub-folder inside
#               the rank directory (e.g. rank1/results/) so inputs and outputs
#               are never mixed.
#
# KEY PARAMETERS OF run_interface()
# -----------------------------------
#   run_mode   : "complex" | "unbound" | "generated"   (default "complex")
#   input_dir  : directory containing the source PDB file(s)
#   results_dir: directory where ALL output files are written
#                  • "complex" / "unbound" : defaults to input_dir
#                  • "generated"           : defaults to <input_dir>/results/
#   pre_split  : {"protein": <path>, "rna": <path>}
#                  Required for "unbound" mode (orchestrator provides paths to
#                  U_pro_PDB and U_RNA_PDB).
#                  Optional override for "generated" mode.
#
# RETURN VALUE
# ------------
#   dict:
#     bsa_complex   (float | "NA")
#     bsa_pro       (float | "NA")
#     bsa_rna       (float | "NA")
#     pro_int       (str)  absolute path to protein .int file
#     rna_int       (str)  absolute path to RNA .int file
#     combined_int  (str)  absolute path to combined .int file
#     has_interface (bool)
#     error         (str | None)
#
# UNCHANGED FROM ORIGINAL
# -----------------------
#   parse_pdbline(), fetch_atomline(), format_values(), the naccess shell
#   calls, the core ASA comparison loop, and the 0.1 Å² cut-off.
# =============================================================================

import os, sys, shutil, time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
timestamp = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
logging.info(f"\n{timestamp}\n###START###")


# ---------------------------------------------------------------------------
# Internal helper — safe path join
# ---------------------------------------------------------------------------
def _p(directory, filename):
    """Return os.path.join(directory, filename)."""
    return os.path.join(directory, filename)


# ---------------------------------------------------------------------------
# UNCHANGED — parse one ATOM line into a 12-element array
# ---------------------------------------------------------------------------
def parse_pdbline(atom_line):
    """Split a PDB ATOM line into its 12 constituent fields."""
    array = ['h'] * 12  # static allotment of 12 variables

    array[0]  = atom_line[:4]            # record type  (ATOM)
    array[1]  = atom_line[4:11].strip()  # atom serial number
    array[2]  = atom_line[11:17].strip() # atom name
    array[3]  = atom_line[17:20].strip() # residue name
    array[4]  = atom_line[20:22].strip() # chain identifier
    array[5]  = atom_line[22:26].strip() # residue sequence number
    array[6]  = atom_line[30:38].strip() # x coordinate
    array[7]  = atom_line[38:46].strip() # y coordinate
    array[8]  = atom_line[46:54].strip() # z coordinate
    array[9]  = atom_line[54:60].strip() # occupancy factor
    array[10] = atom_line[60:66].strip() # temperature factor
    array[11] = atom_line[66:78].strip() # segment / element

    return array


# ---------------------------------------------------------------------------
# UNCHANGED — extract all ATOM lines for one chain from a PDB file
# ---------------------------------------------------------------------------
def fetch_atomline(pdbname, chain):
    """
    Return a string of all ATOM lines belonging to *chain* in *pdbname*.
    Exits the program if the chain is not found (original behaviour).
    """
    string_array = ""
    hit = 0
    for line in open(pdbname):
        if line != "\n" and line != "" and line[:4] == "ATOM":
            split_line = parse_pdbline(line.rstrip())
            if split_line[4] == chain:
                string_array = string_array + line
                hit = 1
            elif hit == 1:
                break
    if hit == 0:
        logging.info(f"\nChain '{chain}' not found in {pdbname} ...Exiting!!!\n")
        exit(0)
    return string_array


# ---------------------------------------------------------------------------
# UNCHANGED — left-pad a numeric string to 6 characters
# ---------------------------------------------------------------------------
def format_values(i):
    """Left-pad *i* to a field width of 6 characters."""
    field = ' ' * 6
    if len(i) < len(field):
        return field[:len(field) - len(i)] + i
    else:
        return i


# ---------------------------------------------------------------------------
# MODIFIED — results_dir parameter replaces working_dir; core logic UNCHANGED
# ---------------------------------------------------------------------------
def generate_interface_atomfile(combinedfile, smallfile, int_filename,
                                results_dir="."):
    """
    Generate a .int file listing interface residues and append to ASA_FINAL.

    Parameters
    ----------
    combinedfile : str   absolute path to the complex .asa file
    smallfile    : str   absolute path to the subunit .asa file
    int_filename : str   absolute path where the output .int file is written
    results_dir  : str   directory where ASA_FINAL summary is appended
    """
    smallfile_lines    = open(smallfile).readlines()
    combinedfile_lines = open(combinedfile).readlines()

    store_line   = ""
    area_subunit = 0.0

    # UNCHANGED comparison logic — cut-off 0.1 Å² difference retained
    for line1 in smallfile_lines:
        if line1 != "\n" and line1 != "":
            subunit_area = float(line1[55:63].strip())
            area_subunit = area_subunit + subunit_area
        for line2 in combinedfile_lines:
            if line1 != "\n" and line2 != "\n" and line1 != "" and line2 != "":
                if line1[:55] == line2[:55]:
                    complex_area = float(line2[55:63].strip())
                    difference   = subunit_area - complex_area
                    if difference > 0.1:   # cut-off value — UNCHANGED
                        store_line = (
                            store_line
                            + line1[:55]
                            + format_values(str.format("{0:.2f}", subunit_area))
                            + format_values(str.format("{0:.2f}", complex_area))
                            + format_values(str.format("{0:.2f}", difference))
                            + "\n"
                        )
                    break

    with open(int_filename, 'w') as outfile:
        outfile.write(store_line)

    # Append ASA total to results_dir/ASA_FINAL
    with open(_p(results_dir, "ASA_FINAL"), 'a') as outfile2:
        outfile2.write(
            os.path.basename(smallfile) + "\t"
            + str.format("{0:.2f}", area_subunit) + "\n"
        )


# ---------------------------------------------------------------------------
# UNCHANGED — sum a column of floats from a .int / .asa file
# ---------------------------------------------------------------------------
def cal_interfacearea(pdbfile, start, end):
    """Return the sum of float values in columns [start:end] of *pdbfile*."""
    try:
        area = 0.0
        for line in open(pdbfile).readlines():
            area = area + float(line[start:end].strip())
        return area
    except FileNotFoundError:
        logging.info(f"{pdbfile} not found for area calculation\n")
        return "NA"


# ---------------------------------------------------------------------------
# MODIFIED — results_dir parameter added
# ---------------------------------------------------------------------------
def check_int_file_empty(filename, results_dir="."):
    """Append the complex stem to INTERFACE_PRESENT or INTERFACE_ABSENT."""
    target_name = os.path.splitext(os.path.basename(filename))[0]
    if os.stat(filename).st_size > 0:
        with open(_p(results_dir, "INTERFACE_PRESENT"), 'a') as out:
            out.write(target_name + "\n")
    else:
        with open(_p(results_dir, "INTERFACE_ABSENT"), 'a') as out:
            out.write(target_name + "\n")


# ---------------------------------------------------------------------------
# Internal helper — consistent failure return value
# ---------------------------------------------------------------------------
def _error_result(pro_path, rna_path, cmplx_path, msg):
    return {
        "bsa_complex":   "NA",
        "bsa_pro":       "NA",
        "bsa_rna":       "NA",
        "pro_int":       os.path.splitext(pro_path)[0] + ".int",
        "rna_int":       os.path.splitext(rna_path)[0] + ".int",
        "combined_int":  os.path.splitext(cmplx_path)[0] + ".int",
        "has_interface": False,
        "error":         msg,
    }


# ---------------------------------------------------------------------------
# MAIN ENGINE
# ---------------------------------------------------------------------------
def run_interface(pdb_file, first_chain, second_chain,
                  run_mode="complex",
                  input_dir=".",
                  results_dir=None,
                  pre_split=None):
    """
    Run naccess-based interface analysis for one protein–RNA chain pair.

    Parameters
    ----------
    pdb_file     : str
        Stem identifier used for naming all output files (e.g. "1ASY").
        Also used to build source PDB paths in "complex" mode.
    first_chain  : str
        Single-letter chain ID for the protein chain.
    second_chain : str
        Single-letter chain ID for the RNA chain.
    run_mode     : {"complex", "unbound", "generated"}, default "complex"
        Selects how input PDB files are located and prepared.
    input_dir    : str, default "."
        Directory containing the source PDB file(s).
          "complex"  : must contain <pdb_file>.pdb
          "unbound"  : orchestrator passes exact paths via pre_split instead
          "generated": rank folder containing protein.pdb and rna.pdb
    results_dir  : str | None, default None
        Directory where ALL output files are written.
          "complex" / "unbound" : defaults to input_dir
          "generated"           : defaults to <input_dir>/results/
    pre_split    : dict | None, default None
        {"protein": <absolute path to protein PDB>,
         "rna":     <absolute path to RNA PDB>}
        Required for "unbound" mode (paths to U_pro_PDB and U_RNA_PDB).
        Optional override for "generated" mode.

    Returns
    -------
    dict
        bsa_complex, bsa_pro, bsa_rna  (float | "NA")
        pro_int, rna_int, combined_int  (str — absolute paths)
        has_interface                   (bool)
        error                           (str | None)
    """

    # ------------------------------------------------------------------
    # 0.  Resolve and create directories
    # ------------------------------------------------------------------
    input_dir = os.path.abspath(input_dir)

    if results_dir is None:
        if run_mode == "generated":
            # REQUIREMENT 2: generated outputs go into <rank_folder>/results/
            results_dir = os.path.join(input_dir, "results")
        else:
            # "complex" and "unbound": write alongside inputs (original behaviour)
            results_dir = input_dir

    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # naccess is run from a staging area so its output files (.asa, .rsa, .log)
    # never land in input_dir; they are moved to results_dir after the run.
    staging_dir = os.path.join(results_dir, "_naccess_staging")
    os.makedirs(staging_dir, exist_ok=True)

    pdb = pdb_file  # stem for all derived filenames

    # Staging paths for the three PDB files naccess will process
    pro_pdb     = _p(staging_dir, pdb + "_" + first_chain + ".pdb")
    RNA_pdb     = _p(staging_dir, pdb + "_" + second_chain + ".pdb")
    complex_pdb = _p(staging_dir, pdb + "_" + first_chain + second_chain + ".pdb")

    pro_stem   = os.path.splitext(pro_pdb)[0]
    rna_stem   = os.path.splitext(RNA_pdb)[0]
    cmplx_stem = os.path.splitext(complex_pdb)[0]

    # ------------------------------------------------------------------
    # 1.  Prepare chain PDB files in staging_dir
    # ------------------------------------------------------------------

    if run_mode == "complex":
        # ----- BOUND COMPLEX (source-truth) mode -------------------------
        # Single combined PDB; extract each chain with fetch_atomline().
        # UNCHANGED logic.
        source_pdb = _p(input_dir, pdb_file + ".pdb")
        logging.info(
            f"[complex] Extracting chains '{first_chain}','{second_chain}' "
            f"from {source_pdb}"
        )
        chain1 = fetch_atomline(source_pdb, first_chain)
        chain2 = fetch_atomline(source_pdb, second_chain)

        with open(pro_pdb, 'w') as f:
            f.write(chain1)
        with open(RNA_pdb, 'w') as f:
            f.write(chain2)
        with open(complex_pdb, 'w') as f:
            f.write(chain1 + chain2)

    elif run_mode == "unbound":
        # ----- UNBOUND mode -----------------------------------------------
        # Protein and RNA come from separate PDB files identified by
        # U_pro_PDB and U_RNA_PDB in PRDBv3.json.  The orchestrator resolves
        # the full file paths and passes them via pre_split:
        #
        #   pre_split = {
        #       "protein": "D:/FFT-scorer/assets/ALL_PDBs/1ASY/1EOV.pdb",
        #       "rna":     "D:/FFT-scorer/assets/ALL_PDBs/1ASY/2TRA.pdb"
        #   }
        #
        # first_chain  = U_PRO_chain  (e.g. "A")
        # second_chain = U_RNA_chain  (e.g. "A")
        if pre_split is None:
            msg = (
                "run_mode='unbound' requires pre_split="
                "{'protein': <U_pro_PDB path>, 'rna': <U_RNA_PDB path>}"
            )
            logging.error(msg)
            return _error_result(pro_pdb, RNA_pdb, complex_pdb, msg)

        pro_source = pre_split["protein"]
        rna_source = pre_split["rna"]
        logging.info(f"[unbound] Protein  : {pro_source}  chain '{first_chain}'")
        logging.info(f"[unbound] RNA      : {rna_source}  chain '{second_chain}'")

        # Extract the relevant chain from each unbound PDB file
        chain1 = fetch_atomline(pro_source, first_chain)
        chain2 = fetch_atomline(rna_source, second_chain)

        with open(pro_pdb, 'w') as f:
            f.write(chain1)
        with open(RNA_pdb, 'w') as f:
            f.write(chain2)
        # Synthesise combined PDB so naccess can compute complex-state ASA
        with open(complex_pdb, 'w') as f:
            f.write(chain1 + chain2)

    elif run_mode == "generated":
        # ----- GENERATED / DOCKED mode ------------------------------------
        # protein.pdb and rna.pdb are already-split files in input_dir.
        # Chain IDs are already embedded; no extraction needed.
        if pre_split is not None:
            pro_source = pre_split["protein"]
            rna_source = pre_split["rna"]
        else:
            # Default: standard filenames produced by the FFT docking code
            pro_source = _p(input_dir, "protein.pdb")
            rna_source = _p(input_dir, "rna.pdb")

        logging.info(f"[generated] Protein: {pro_source}")
        logging.info(f"[generated] RNA    : {rna_source}")

        shutil.copy2(pro_source, pro_pdb)
        shutil.copy2(rna_source, RNA_pdb)

        # Build combined PDB by concatenating the two files
        with open(complex_pdb, 'w') as cmplx_f:
            for src in [pro_source, rna_source]:
                with open(src) as src_f:
                    cmplx_f.write(src_f.read())

    else:
        msg = (
            f"Unknown run_mode '{run_mode}'. "
            f"Must be 'complex', 'unbound', or 'generated'."
        )
        logging.error(msg)
        return _error_result(pro_pdb, RNA_pdb, complex_pdb, msg)

    # ------------------------------------------------------------------
    # 2.  Run naccess from staging_dir
    #     naccess writes .asa / .rsa / .log files next to the input PDB.
    # ------------------------------------------------------------------
    original_dir = os.getcwd()
    os.chdir(staging_dir)

    logging.info("::naccess running::")
    os.system("naccess " + os.path.basename(pro_pdb))
    os.system("naccess " + os.path.basename(RNA_pdb))
    os.system("naccess " + os.path.basename(complex_pdb))
    logging.info("::naccess done::")

    os.chdir(original_dir)   # always restore the caller's directory

    # ------------------------------------------------------------------
    # 3.  Locate naccess .asa outputs (all in staging_dir)
    # ------------------------------------------------------------------
    pro_asa   = pro_stem   + ".asa"
    rna_asa   = rna_stem   + ".asa"
    cmplx_asa = cmplx_stem + ".asa"

    # Final .int files go directly into results_dir
    pro_int_path      = _p(results_dir, pdb + "_" + first_chain + ".int")
    rna_int_path      = _p(results_dir, pdb + "_" + second_chain + ".int")
    combined_int_path = _p(results_dir, pdb + "_" + first_chain + second_chain + ".int")

    result = {
        "bsa_complex":   "NA",
        "bsa_pro":       "NA",
        "bsa_rna":       "NA",
        "pro_int":       os.path.abspath(pro_int_path),
        "rna_int":       os.path.abspath(rna_int_path),
        "combined_int":  os.path.abspath(combined_int_path),
        "has_interface": False,
        "error":         None,
    }

    # ------------------------------------------------------------------
    # 4.  Generate .int files and compute BSA — UNCHANGED logic
    # ------------------------------------------------------------------
    if (os.path.exists(pro_asa)
            and os.path.exists(rna_asa)
            and os.path.exists(cmplx_asa)):

        generate_interface_atomfile(cmplx_asa, pro_asa, pro_int_path,  results_dir)
        generate_interface_atomfile(cmplx_asa, rna_asa, rna_int_path,  results_dir)

        bsa_pro = cal_interfacearea(pro_int_path, 67, 73)
        bsa_rna = cal_interfacearea(rna_int_path, 67, 73)
        bsa_complex = (bsa_pro + bsa_rna) if (bsa_pro != "NA" and bsa_rna != "NA") else "NA"

        # Write BSA summary to results_dir/BSA_FINAL
        with open(_p(results_dir, "BSA_FINAL"), 'a') as sasa_summary:
            sasa_summary.write(
                pdb.ljust(10, ' ')
                + "Complex".rjust(10, ' ')
                + "Protein".rjust(10, ' ')
                + "RNA".rjust(10, ' ')
                + "\n"
            )
            if bsa_complex != "NA":
                sasa_summary.write(
                    "BSA ".ljust(10, ' ')
                    + ("%.1f" % bsa_complex).rjust(10, ' ')
                    + ("%.1f" % bsa_pro).rjust(10, ' ')
                    + ("%.1f" % bsa_rna).rjust(10, ' ')
                    + '\n'
                )

        # Combine protein and RNA .int files — replaces os.system("cat...")
        with open(combined_int_path, 'w') as combined_f:
            for int_path in [pro_int_path, rna_int_path]:
                if os.path.exists(int_path):
                    with open(int_path) as part_f:
                        combined_f.write(part_f.read())

        check_int_file_empty(combined_int_path, results_dir)
        has_interface = os.stat(combined_int_path).st_size > 0

        # Move all naccess intermediate files from staging into results_dir
        for fname in os.listdir(staging_dir):
            fpath = _p(staging_dir, fname)
            if os.path.isfile(fpath):
                dest = _p(results_dir, fname)
                if os.path.exists(dest):
                    os.remove(dest)
                shutil.move(fpath, dest)
        shutil.rmtree(staging_dir, ignore_errors=True)

        result.update({
            "bsa_complex":   bsa_complex,
            "bsa_pro":       bsa_pro,
            "bsa_rna":       bsa_rna,
            "pro_int":       os.path.abspath(pro_int_path),
            "rna_int":       os.path.abspath(rna_int_path),
            "combined_int":  os.path.abspath(combined_int_path),
            "has_interface": has_interface,
            "error":         None,
        })

    else:
        missing = [p for p in [pro_asa, rna_asa, cmplx_asa]
                   if not os.path.exists(p)]
        msg = f"naccess .asa file(s) not found: {missing}"
        logging.error(msg)
        result["error"] = msg
        shutil.rmtree(staging_dir, ignore_errors=True)

    logging.info(f"###DONE###\n{timestamp}\n")
    return result


# ---------------------------------------------------------------------------
# MODIFIED — main() accepts optional arguments; List_1.txt logic UNCHANGED
# ---------------------------------------------------------------------------
def main(input_file=None, working_dir="."):
    """
    Entry point for standalone / legacy usage (List_1.txt driven).

    Parameters
    ----------
    input_file : str | None
        Path to the List_1.txt input file.  Defaults to List_1.txt inside
        working_dir (original behaviour).
    working_dir : str
        Root directory for input PDB files and output summary files.
        Defaults to "." (original behaviour).
    """
    if input_file is None:
        input_file = _p(working_dir, "List_1.txt")

    # UNCHANGED — remove stale summary files before a fresh run
    remove_files = ["BSA_FINAL", "ASA_FINAL", "INTERFACE_PRESENT", "INTERFACE_ABSENT"]
    for fname in remove_files:
        fpath = _p(working_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            logging.info(f"Removed: {fpath}")

    with open(input_file, "r") as infile:
        # UNCHANGED — List_1.txt format: {1A02\tNFJ:AB}
        for line in infile:
            line = line.strip()
            if not line:
                continue
            complex_pdb  = line.split()[0]
            chains       = line.split()[1]
            first_chain  = set(chains.split(':')[0])
            second_chain = set(chains.split(':')[1])

            for i in first_chain:
                for j in second_chain:
                    logging.info(
                        f"Running program for First chain:{i} with second chain:{j}"
                    )
                    result = run_interface(
                        pdb_file=complex_pdb,
                        first_chain=i,
                        second_chain=j,
                        run_mode="complex",
                        input_dir=working_dir,
                        results_dir=working_dir,
                    )
                    logging.info(
                        f"Calculation done for chains: {(i, j)} "
                        f"| BSA={result['bsa_complex']}"
                    )


if __name__ == "__main__":
    main()