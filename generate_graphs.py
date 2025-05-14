import torch
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

file_path = input("Please enter the path to the .xyz file: ")

def read_xyz_multi(file_path):
    """

    Read an .xyz file that contains multiple molecular structures.
    
    Returns:
        - List of (atoms, coordinates) for each structure.
    """
    structures = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Skip empty lines or comment lines (lines that don't have an integer atom count)
        line = lines[i].strip()
        
        # Skip lines that are empty or not a valid number
        if not line or not line.isdigit():
            print(f"Skipping invalid or empty line: {line}")
            i += 1
            continue
        
        # Read number of atoms
        try:
            num_atoms = int(line)  # This should now correctly read the number of atoms
        except ValueError:
            print(f"Skipping invalid line for atom count: {line}")
            i += 1
            continue

        i += 2  # Skip title/comment line
        
        atoms = []
        coordinates = []
        
        for _ in range(num_atoms):
            line = lines[i].strip()
            
            # Skip any line that doesn't have the expected number of values
            if len(line.split()) < 4:  # We expect at least atom type and 3 coordinates
                print(f"Skipping invalid line: {line}")
                i += 1
                continue
            
            # Correct any extra spaces by splitting and cleaning up the line
            line_parts = line.split()
            atom_type = line_parts[0]
            try:
                x, y, z = map(float, line_parts[1:])
            except ValueError:
                print(f"Skipping line with invalid coordinates: {line}")
                i += 1
                continue
            
            atoms.append(atom_type)
            coordinates.append([x, y, z])
            i += 1  # Move to next line

        structures.append((atoms, np.array(coordinates)))  # Store structure

    return structures

def generate_graph(atoms, coordinates):
    """
    Convert atomic data to graph representation.
    
    Returns:
        - PyTorch Geometric Data object.
    """
    # Convert atom types to unique indices
    atom_types = set(atoms)
    atom_type_dict = {atom: idx for idx, atom in enumerate(atom_types)}
    atom_indices = [atom_type_dict[atom] for atom in atoms]

    # Convert coordinates to tensor
    coords_tensor = torch.tensor(coordinates, dtype=torch.float)
    
    # Create adjacency matrix (graph connections)
    num_atoms = len(atoms)
    edge_index = []

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            if dist < 2.0:  # Threshold for bond detection
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected graph

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Convert atom types to tensor
    atom_types_tensor = torch.tensor(atom_indices, dtype=torch.long)

    # Return the PyTorch Geometric Data object
    return Data(x=atom_types_tensor, edge_index=edge_index, pos=coords_tensor)

structures = read_xyz_multi(file_path)

# Generate graphs for all structures
graph_data_list = [generate_graph(atoms, coords) for atoms, coords in structures]


def read_xyz_multi(file_path):
    """
    Read an .xyz file that contains multiple molecular structures.
    
    Returns:
        - List of (atoms, coordinates) for each structure.
    """
    structures = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())  # Read number of atoms
        i += 1  # Skip title line
        
        atoms = []
        coordinates = []
        
        for _ in range(num_atoms):
            line = lines[i].strip().split()
            atom_type = line[0]
            x, y, z = map(float, line[1:])
            
            atoms.append(atom_type)
            coordinates.append([x, y, z])
            i += 1  # Move to next line

        structures.append((atoms, np.array(coordinates)))  # Store structure

    return structures

def generate_graph(atoms, coordinates):
    """
    Convert atomic data to graph representation.
    
    Returns:
        - PyTorch Geometric Data object.
    """
    # Convert atom types to unique indices
    atom_types = set(atoms)
    atom_type_dict = {atom: idx for idx, atom in enumerate(atom_types)}
    atom_indices = [atom_type_dict[atom] for atom in atoms]

    # Convert coordinates to tensor
    coords_tensor = torch.tensor(coordinates, dtype=torch.float)
    
    # Create adjacency matrix (graph connections)
    num_atoms = len(atoms)
    edge_index = []

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            if dist < 2.0:  # Threshold for bond detection
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected graph

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Convert atom types to tensor
    atom_types_tensor = torch.tensor(atom_indices, dtype=torch.long)

    # Return the PyTorch Geometric Data object
    return Data(x=atom_types_tensor, edge_index=edge_index, pos=coords_tensor)

# Generate graphs for all structures
graph_data_list = [generate_graph(atoms, coords) for atoms, coords in structures]

# Print each graph data
for idx, graph in enumerate(graph_data_list):
    print(f"Structure {idx + 1}:")
    print(graph)
    print()

# Save graph data
with open("graph_data.pkl", "wb") as f:
    pickle.dump(graph_data_list, f)

# Save structure data
with open("structure_data.pkl", "wb") as f:
    pickle.dump(structures, f)

print("Graph data saved successfully.")
