# Molecular Graph Generator

This script processes `.xyz` files containing multiple molecular structures and converts them into graph representations using **PyTorch Geometric**.

## Features

- Parses `.xyz` files with multiple molecules
- Converts each structure into a graph:
  - **Nodes**: atoms (indexed)
  - **Edges**: bonds inferred by distance (< 2.0 Å)
- Saves graphs and raw structures as `.pkl` files

## Requirements

- Python 3.7+
- PyTorch
- PyTorch Geometric
- NumPy
- RDKit

## Output
- graph_data.pkl: list of PyTorch Geometric Data objects
- structure_data.pkl: list of (atoms, coordinates)

## Notes
Bonds are inferred using a 2.0 Å threshold
The parser handles minor formatting issues in .xyz files
