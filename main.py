import torch
import numpy as np
import networkx as nx
import gudhi as gd
 
'''
Input takes tensors but we can easily turn a graph into a pytorch tensor by 
matrix = nx.to_numpy_array(G,weight'weight',noedge=float('inf'))
tensor = torch.tenspr(matrix, dtype=torch.float32)
with better variable names
'''
def PDFromGraph(adj_tensor,max_dimension,hom_dims=2): 
    diagrams = []

    # Stop tracking adjacency matrix
    adj_matrix = adj_tensor.detach().numpy() 

    # Everythign here is the same
    rips_complex = gd.RipsComplex(distance_matrix=adj_matrix)
    st = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    st.compute_persistence()
    
    # Isolate the vertices that caused the birth or death of our persistence
    generators = st.flag_persistence_generators()

    for i in range(hom_dims):
        # Gudhi is weird
        if i == 0:
           generators_i = generators[i]
        else:
           generators_i = generators[1][i-1]
        if len(generators_i) == 0:
            diagrams.append(torch.empty((0,2)))
            break

        hi_gens = torch.tensor(generators_i)

        # Torch will now track the ones that caused the birth and death and backprop along those
        if i == 0:
            birth_values = adj_tensor[hi_gens[:,0], hi_gens[:,0]]
            death_values = adj_tensor[hi_gens[:,1], hi_gens[:,2]]
        else:
            birth_values = adj_tensor[hi_gens[:,0], hi_gens[:,1]]
            death_values = adj_tensor[hi_gens[:,2], hi_gens[:,3]]
        diagram = torch.stack([birth_values,death_values], dim=-1)
        diagrams.append(diagram)
    return diagrams
