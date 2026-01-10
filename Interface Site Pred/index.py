import os
import json
import pandas as pd
import numpy as np

# 1. CONFIGURATION
BASE_DIR = ""
JSON_PATH = ""

# 2. LOAD METADATA
print(f"Loading metadata from {JSON_PATH}...")
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Total metadata entries: {len(df)}")

# 3. ROBUST PATH CONSTRUCTION (3-COLUMN LOGIC)
def get_all_paths(row):
  
    c_id = row.get('C_PDB')      # e.g., "1ASY"
    u_pro_id = row.get('U_pro_PDB') # e.g., "1EOV"
    u_rna_id = row.get('U_RNA_PDB') # e.g., "2TRA"
    
    # Base folder: /kaggle/.../1ASY
    complex_dir = os.path.join(BASE_DIR, c_id)
    
  
    # PATH 1: BOUND COMPLEX (The Anchor)
    bound_path = os.path.join(complex_dir, f"{c_id}.pdb")
    if not os.path.exists(bound_path):
        return None  # If bound complex is missing, the whole entry is invalid
    
    # PATH 2: UNBOUND PROTEIN
    u_pro_path = None
    if u_pro_id and isinstance(u_pro_id, str):
        clean_p_id = u_pro_id.split(':')[0].strip() # Clean "1EOV:A" -> "1EOV"
        cand_p_path = os.path.join(complex_dir, f"{clean_p_id}.pdb")
        if os.path.exists(cand_p_path):
            u_pro_path = cand_p_path
            
    # PATH 3: UNBOUND RNA
    u_rna_path = None
    if u_rna_id and isinstance(u_rna_id, str):
        clean_r_id = u_rna_id.split(':')[0].strip() # Clean "2TRA:A" -> "2TRA"
        cand_r_path = os.path.join(complex_dir, f"{clean_r_id}.pdb")
        if os.path.exists(cand_r_path):
            u_rna_path = cand_r_path

    # Return structured data
    return {
        'bound_path': bound_path,
        'unbound_pro_path': u_pro_path,
        'unbound_rna_path': u_rna_path
    }

# 4. EXECUTION
print("Verifying files on disk (processing 3-column paths)...")

results = []
for idx, row in df.iterrows():
    paths = get_all_paths(row)
    
    if paths: # Only proceed if Bound path exists
        results.append({
            'pdb_id': row['C_PDB'],
            'docking_case': row['Docking_case'],
            
            # FILE PATHS
            'bound_path': paths['bound_path'],
            'unbound_pro_path': paths['unbound_pro_path'], # Can be None
            'unbound_rna_path': paths['unbound_rna_path'], # Can be None
            
            # CHAINS (Critical for parsing)
            'bound_pro_chain': row.get('C_pro_chain'),
            'unbound_pro_chain': row.get('U_PRO_chain'),
            'bound_rna_chain': row.get('C_RNA_chain'),
            'unbound_rna_chain': row.get('U_RNA_chain'),
            
            # METADATA
            'status': 'Success'
        })
    else:
        # Record broken entries if needed, or skip
        pass

# Create Master DataFrame
master_df = pd.DataFrame(results)

# 5. DIAGNOSTICS & EXPORT
print("-" * 40)
print("INDEX GENERATION COMPLETE")
print("-" * 40)
print(f"Total Verified Entries:  {len(master_df)}")
print("-" * 40)
print("Data Availability Breakdown:")
print(f"Entries with Unbound PROTEIN: {master_df['unbound_pro_path'].notnull().sum()}")
print(f"Entries with Unbound RNA:     {master_df['unbound_rna_path'].notnull().sum()}")
print(f"Entries with BOTH Unbound:    {(master_df['unbound_pro_path'].notnull()) & (master_df['unbound_rna_path'].notnull()).sum()}")
print("-" * 40)

# Save
output_filename = "master_index.csv"
master_df.to_csv(output_filename, index=False)

print(f"Saved to: {output_filename}")
print("\nFirst 3 rows (Transposed for readability):")
print(master_df.head(3).T)
