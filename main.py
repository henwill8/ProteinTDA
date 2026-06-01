import numpy as np
import networkx as nx
import gudhi as gd
 
def PDFromGraph(G,max_dimension): 
    adj_matrix = nx.to_numpy_array(G, weight='weight', nonedge=float('inf'))
    rips_complex = gd.RipsComplex(distance_matrix=adj_matrix)
    st = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    return st.persistence()

