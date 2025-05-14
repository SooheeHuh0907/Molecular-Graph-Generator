
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from rdkit import Chem
from rdkit.Chem import Draw

# Load graph data
with open("graph_data.pkl", "rb") as f:
    graph_data_list = pickle.load(f)

# Load molecular structure data
with open("structure_data.pkl", "rb") as f:
    structures = pickle.load(f)

def visualize_graph(graph, index):
    """
    Convert PyTorch Geometric Data to NetworkX graph and plot it.
    """
    G = to_networkx(graph, to_undirected=True)  # Convert PyG to NetworkX
    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title(f"Graph Representation of Structure {index + 1}")
    plt.show()

def visualize_all_graphs(graph_data_list):
    """
    Visualize all graph structures in separate plots.
    """
    for i, graph in enumerate(graph_data_list):
        visualize_graph(graph, i)

# Ask the user what they want to do
choice = input("Enter '1' to visualize a single graph, '2' to visualize all graphs: ")

if choice == '1':
    index = int(input("Enter the index of the graph you want to visualize: "))
    visualize_graph(graph_data_list[index], index)
elif choice == '2':
    visualize_all_graphs(graph_data_list)
else:
    print("Invalid choice.")
