# Molecular Graph Generator

generate_graphs.py # Parses .xyz files and generates molecular graphs
visualize_graph.py # Visualizes molecular graph structures (2D)
visualize_3D.py # Visualizes molecular structures in 3D space
graph_data.pkl # (Generated) Graph representations of molecules
structure_data.pkl # (Generated) Atom types and 3D coordinates

## Requirements

- Python 3.7+
- PyTorch
- PyTorch Geometric
- NumPy
- RDKit
- Matplotlib
- NetworkX


# generate_graphs.py 
This script processes `.xyz` files containing multiple molecular structures and converts them into graph representations using **PyTorch Geometric**.

## Features

- Parses `.xyz` files with multiple molecules
- Converts each structure into a graph:
  - **Nodes**: atoms (indexed)
  - **Edges**: bonds inferred by distance (< 2.0 Å)
- Saves graphs and raw structures as `.pkl` files

## Output
- graph_data.pkl: list of PyTorch Geometric Data objects
- structure_data.pkl: list of (atoms, coordinates)

## Notes
Bonds are inferred using a 2.0 Å threshold
The parser handles minor formatting issues in .xyz files

# visualize_graph.py
Display molecular graphs with node/edge structure using NetworkX.

Option 1: visualize a single graph

Option 2: visualize all graphs in the dataset


# visualize_3D.py
Render molecules in 3D space with atoms, bonds, and bond lengths.

Option 1: visualize a single molecule

Option 2: visualize all molecules sequentially
