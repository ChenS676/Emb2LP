
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import networkx as nx
# import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import from_networkx
import numpy as np
# from matplotlib import cm
import torch.nn as nn

class GraphGeneration():
    def __init__(self, m, n, emb_dim, graph_type, heterophily=False, homophily=False):
        self.m = m
        self.n = n
        self.emb_dim = emb_dim
        self.graph_type = graph_type
        self.heterophily = heterophily
        self.homophily = homophily
        
    def create_kagome_lattice(self):
        """ Create a Kagome lattice and return its NetworkX graph and positions. """
        G = nx.Graph()
        pos = {}
        
        def node_id(x, y, offset):
            return 2 * (x * self.n + y) + offset
        
        for x in range(self.m):
            for y in range(self.n):
                # Two nodes per cell (offset 0 and 1)
                current_id0 = node_id(x, y, 0)
                current_id1 = node_id(x, y, 1)
                pos[current_id0] = (y, x)
                pos[current_id1] = (y + 0.5, x + 0.5)
                
                # Add nodes
                G.add_node(current_id0)
                G.add_node(current_id1)
                
                # Right and down connections
                if y < self.n - 1:
                    right_id0 = node_id(x, y + 1, 0)
                    right_id1 = node_id(x, y + 1, 1)
                    G.add_edge(current_id0, right_id0)
                    G.add_edge(right_id1, right_id0)
                    G.add_edge(right_id0, current_id0)
                    G.add_edge(right_id0, right_id1)
                    
                if x < self.m - 1:
                    down_id0 = node_id(x + 1, y, 0)
                    down_id1 = node_id(x + 1, y, 1)
                    G.add_edge(current_id0, down_id0)
                    G.add_edge(current_id1, down_id1)
                    G.add_edge(down_id0, current_id0)
                    G.add_edge(down_id1, current_id1)
                
                # Diagonal connections
                if x < self.m - 1 and y < self.n - 1:
                    diag_id0 = node_id(x + 1, y + 1, 0)
                    diag_id1 = node_id(x + 1, y + 1, 1)
                    G.add_edge(current_id1, diag_id0)
                    G.add_edge(diag_id0, current_id1)
                    G.add_edge(current_id1, diag_id1)
                    G.add_edge(diag_id1, current_id1)
        
        return G, pos

    def create_square_grid(self):
        num_nodes = self.m * self.n
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        
        def node_id(x, y):
            return x * self.n + y
        
        for x in range(self.m):
            for y in range(self.n):
                current_id = node_id(x, y)
                
                # Right neighbor
                if y < self.n - 1:
                    right_id = node_id(x, y + 1)
                    adj_matrix[current_id, right_id] = 1
                    adj_matrix[right_id, current_id] = 1
                    
                # Down neighbor
                if x < self.m - 1:
                    down_id = node_id(x + 1, y)
                    adj_matrix[current_id, down_id] = 1
                    adj_matrix[down_id, current_id] = 1

        # Generate positions for visualization
        pos = {(x * self.n + y): (y, x) for x in range(self.m) for y in range(self.n)}

        # Create NetworkX graph from adjacency matrix
        G = nx.from_numpy_array(adj_matrix)
        
        # Assign labels in a checkerboard pattern
        if self.heterophily:
            for x in range(self.m):
                for y in range(self.n):
                    current_id = node_id(x, y)
                    G.nodes[current_id]['label'] = (x + y) % 2
        
        if self.homophily:
            for x in range(self.m):
                for y in range(self.n):
                    current_id = node_id(x, y)
                    if current_id >= len(G.nodes) / 2:
                        G.nodes[current_id]['label'] = 1
                    else:
                        G.nodes[current_id]['label'] = 0
        return G, pos

    def create_grid_graph(self):
        """ Create a grid graph and return its NetworkX graph and positions. """
        G = nx.grid_2d_graph(self.m, self.n)
        pos = {(x, y): (x, y) for x, y in G.nodes()}
        return G, pos

    def create_triangle_grid(self):
        G = nx.triangular_lattice_graph(self.m, self.n)
        pos =  nx.get_node_attributes(G, 'pos')
        return G, pos

    def create_hexagonal_grid(self):
        # Generate the hexagonal lattice graph
        G = nx.hexagonal_lattice_graph(self.m, self.n)
        pos = nx.get_node_attributes(G, 'pos')
        
        if self.heterophily:
            for node in G.nodes:
                id1, id2 = node
                G.nodes[node]['label'] = (id1 + id2) % 2
        
        if self.homophily:
            for node in G.nodes:
                x, _ = node
                if x >= (max(self.m, self.n) / 2):
                    G.nodes[node]['label'] = 1
                else:
                    G.nodes[node]['label'] = 0
            
            # Solves the problem with painting in the center of the network
            if self.m % 2 == 0 or self.n % 2 == 0:
                x = max(self.m, self.n) // 2
                y = 0
                while y < ((self.m * 2 + 1) / 2):
                    node = (x, y)
                    if self.m < self.n:
                        G.nodes[node]['label'] = 0
                    else:
                        G.nodes[node]['label'] = 1
                    y += 1

        return G, pos

    def generate_graph(self):
        """
        Generates a PyG graph for nodes in a NetworkX graph based on their positions.

        Parameters:
        m (int): Number of rows in the grid graph.
        n (int): Number of columns in the grid graph.
        emb_dim (int): The dimension of the embeddings.

        Returns:
        Data: The generated graph with embeddings.
        """

        if self.graph_type == 'grid':
            G, pos = self.create_grid_graph()
        elif self.graph_type == 'square_grid':
            G, pos = self.create_square_grid()
        elif self.graph_type == 'triangle':
            G, pos = self.create_triangle_grid()
        elif self.graph_type == 'hexagonal':
            G, pos = self.create_hexagonal_grid()
        elif self.graph_type == 'kagome':
            G, pos = self.create_kagome_lattice()
        else:
            raise ValueError(f"Invalid graph type: {self.graph_type}")
        
        # TODO: CHECK
        data = from_networkx(G)
        if self.heterophily or self.homophily:
            labels = [G.nodes[node]['label'] for node in G.nodes]
            emb_layer = nn.Embedding(2, self.emb_dim)  # 2 classes for checkerboard pattern
            with torch.no_grad():
                label_tensor = torch.tensor(labels, dtype=torch.int64)
                vectors = emb_layer(label_tensor)
            data.x = vectors    
        else:
            emb_layer = nn.Embedding(data.num_nodes, self.emb_dim)
        
            pos_list = []
            for _, x in pos.items():
                pos_list.append(x[0] * self.m + x[1])
            
            pos_array = np.asarray(pos_list)
            pos_array = np.clip(pos_array, 0, data.num_nodes - 1)
            with torch.no_grad():
                pos_tensor = torch.tensor(pos_array, dtype=torch.int64)
                vectors = emb_layer(pos_tensor)
            data.x = vectors

        return data, G, pos
