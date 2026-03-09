""" This is a Kaggle native code and is therefore not suitable for local usage"""


import pandas as pd
import plotly.graph_objects as go
import os
import math

# 1. CONFIGURATION FOR VISUALIZATION
# Define colors and sizes (radii) based on atom name or element.
# We prioritize exact atom names (like 'CA'), then fallback to elements.
ATOM_PROPERTIES = {
    # Specific Atoms
    'CA': {'color': '#A9A9A9', 'size': 1.2},  # Carbon-alpha (smaller, dark grey)
    'P':  {'color': '#FFA500', 'size': 2.0},  # Phosphorus in RNA backbone (larger, orange)
    
    # General Elements
    'C': {'color': '#C8C8C8', 'size': 1.7},   # Carbon (light grey)
    'N': {'color': '#0000FF', 'size': 1.55},  # Nitrogen (blue, larger than CA)
    'O': {'color': '#FF0000', 'size': 1.52},  # Oxygen (red)
    'S': {'color': '#FFFF00', 'size': 1.8},   # Sulfur (yellow)
    'H': {'color': '#FFFFFF', 'size': 1.2},   # Hydrogen (white)
    
    # Default fallback
    'DEFAULT': {'color': '#32CD32', 'size': 1.5} # Unknown atoms (green)
}

def get_atom_props(atom_name, element):
    """Retrieve color and size based on atom name or element."""
    if atom_name in ATOM_PROPERTIES:
        return ATOM_PROPERTIES[atom_name]
    elif element in ATOM_PROPERTIES:
        return ATOM_PROPERTIES[element]
    else:
        return ATOM_PROPERTIES['DEFAULT']

# 2. PDB PARSER
def parse_pdb(filepath):
    """Extract coordinates, atom name, and element from a PDB file."""
    if not os.path.exists(filepath):
        print(f"Warning: The file {filepath} does not exist on this system.")
        return []
        
    atoms = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # PDB fixed-width format parsing
                atom_name = line[12:16].strip()
                element = line[76:78].strip()
                
                # If element column is empty, guess from atom name
                if not element:
                    element = ''.join([c for c in atom_name if c.isalpha()])[0]
                    
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                atoms.append({
                    'name': atom_name,
                    'element': element,
                    'x': x, 'y': y, 'z': z
                })
    return atoms

# 3. VISUALIZATION ENGINE
def visualize_atoms(atoms, title="3D Molecular Visualization"):
    """Plot atoms using Plotly."""
    if not atoms:
        print("No atoms to visualize.")
        return
        
    x_vals, y_vals, z_vals = [], [], []
    colors, sizes, hover_texts = [], [], []
    
    for atom in atoms:
        x_vals.append(atom['x'])
        y_vals.append(atom['y'])
        z_vals.append(atom['z'])
        
        props = get_atom_props(atom['name'], atom['element'])
        colors.append(props['color'])
        sizes.append(props['size'] * 5) # Scale factor for Plotly marker size
        
        hover_texts.append(f"Atom: {atom['name']}<br>Element: {atom['element']}<br>X: {atom['x']}<br>Y: {atom['y']}<br>Z: {atom['z']}")
        
    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.9,
            line=dict(width=0.5, color='black') # Adds a slight border to spheres for depth
        ),
        text=hover_texts,
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='data' # Ensures accurate 3D proportions
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

# 4. MAIN INTERACTIVE PROGRAM
def main():
    # 1. Load the master index
    csv_path = '/kaggle/input/datasets/shuvamvidyarthy/master-index/master_index.csv'
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please upload it to your Kaggle working directory.")
        return
    
    # 2. Prompt for PDB ID
    pdb_id = input("Enter PDB ID (e.g., 1ASY, 1B23): ").strip().upper()
    
    # Filter the dataframe
    row = df[df['pdb_id'] == pdb_id]
    if row.empty:
        print(f"Error: PDB ID '{pdb_id}' not found in the master index.")
        return
    
    # 3. Extract paths and fix Kaggle directory structure
    raw_pro_path = row['unbound_pro_path'].values[0]
    raw_rna_path = row['unbound_rna_path'].values[0]
    
    # Helper function to inject the missing dataset path
    def fix_kaggle_path(path):
        if pd.notna(path) and str(path).strip() != "":
            return str(path).replace("/kaggle/input/", "/kaggle/input/datasets/shuvamvidyarthy/")
        return path
    
    pro_path = fix_kaggle_path(raw_pro_path)
    rna_path = fix_kaggle_path(raw_rna_path)
    
    # Check if paths are actually valid strings and not empty
    has_pro = pd.notna(pro_path) and str(pro_path).strip() != ""
    has_rna = pd.notna(rna_path) and str(rna_path).strip() != ""
    
    # 4. Prompt for viewing choice based on availability
    if has_rna:
        choice = input(f"For {pdb_id}, what would you like to view? (Protein / RNA / Both): ").strip().lower()
    else:
        print(f"\nError: Unbound RNA .pdb file not found for {pdb_id}.")
        choice = input("Would you like to view the UB Protein file instead? (Yes / No): ").strip().lower()
        if choice in ['yes', 'y']:
            choice = 'protein'
        else:
            print("Visualization aborted.")
            return

    # 5. Load the requested files
    atoms_to_plot = []
    title_parts = []
    
    if choice in ['protein', 'both']:
        if has_pro:
            print(f"Loading Protein from: {pro_path}")
            atoms_to_plot.extend(parse_pdb(pro_path))
            title_parts.append("UB Protein")
        else:
            print("Error: Protein path is missing in the dataset.")
            
    if choice in ['rna', 'both']:
        if has_rna:
            print(f"Loading RNA from: {rna_path}")
            atoms_to_plot.extend(parse_pdb(rna_path))
            title_parts.append("UB RNA")
        else:
            # This block is a fallback, logic shouldn't technically reach here due to earlier checks
            print("Error: RNA path is missing.")
            
    if not atoms_to_plot:
        print("No valid structures to display.")
        return
        
    # 6. Visualize
    final_title = f"{pdb_id} - {' & '.join(title_parts)}"
    print("Generating 3D Visualization...")
    visualize_atoms(atoms_to_plot, title=final_title)

# Execute the program
if __name__ == "__main__":
    main()
