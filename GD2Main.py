import math
import sys
import time
import os
import random

import numpy as np
from scipy.optimize import minimize

#NetworkX
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_agraph import read_dot as nx_read_dot

#Metrics
import ksymmetry
import crossings
import stress
import neighbors_preservation
import uniformity_edge_length
#import areafunctions

def scale_graph(G, alpha):

    H = G.copy()

    for currVStr in nx.nodes(H):

        currV = H.node[currVStr]

        x = float(currV['pos'].split(",")[0])
        y = float(currV['pos'].split(",")[1])

        x = x * alpha
        y = y * alpha

        currV['pos'] = str(x)+","+str(y)

    return H

def normalize_edge_length(G):

    shortest_edge = float("inf")

    vertices_positions = nx.get_node_attributes(G, "pos")

    for e in nx.edges(G):

        (u,v) = e

        x_v = float(vertices_positions[v].split(",")[0])
        y_v = float(vertices_positions[v].split(",")[1])

        x_u = float(vertices_positions[u].split(",")[0])
        y_u = float(vertices_positions[u].split(",")[1])

        curr_length = math.sqrt((x_v-x_u)**2+(y_v-y_u)**2)

        shortest_edge = min(shortest_edge, curr_length)


    print(shortest_edge)

    alpha = 1/shortest_edge

    G = scale_graph(G, alpha)

    return G




def writeSPXPositiontoNetworkXGraph(G, X):
    '''
    Convert matrix X to NetworkX graph structure
    '''
    positions = dict()
    sorted_v = sorted(nx.nodes(G))
    for i in range(0, len(sorted_v)):
        v = sorted_v[i]
        x = X[i,:][0]
        y = X[i,:][1]
        v_pos = str(x)+","+str(y)
        positions[v] = v_pos
    nx.set_node_attributes(G, positions, 'pos')
    return G


def netoworkxPositionsToMatrix(G):
    '''
        Convert NetwokX pos to Matrix
    '''

    n = nx.number_of_nodes(G)
    X_curr = np.random.rand(n,2)*100 - 50
    vertices_positions = nx.get_node_attributes(G, "pos")
    nodes_list_sorted = sorted(nx.nodes(G))

    for i in range(0, len(nodes_list_sorted)):
        curr_n_id = nodes_list_sorted[i]
        x = float(vertices_positions[curr_n_id].split(",")[0])
        y = float(vertices_positions[curr_n_id].split(",")[1])
        tmp = np.zeros((2))
        tmp[0] = x
        tmp[1] = y
        X_curr[i] = tmp

    return X_curr


def computeGraphDistances(G):
    '''
        Computes all pairs shortest paths on the given graph.
    '''

    G_undirected = nx.Graph(G)
    distances = nx.floyd_warshall(G_undirected)
    return distances

def printMetrics(G):
    '''
        Set inital values before optimization
    '''

    global initial_st
    global all_pairs_sp
    global initial_cr
    global initial_ar
    global initial_asp
    # Do some preliminary stuff

    # Stress will be normalized considering the first value as max
    # To speed up ST precompute all pairs shortest paths
    initial_st = 1
    if compute_st:
        initial_st = stress.stress(G, all_sp=all_pairs_sp)
        print("ST:", initial_st, end=" - ")
        if initial_st == 0:
            initial_st = 1
        if all_pairs_sp is None:
            all_pairs_sp = nx.shortest_path(G)

    # To speed up NP precompute all pairs shortest paths
    if compute_np:
        if all_pairs_sp is None:
            all_pairs_sp = nx.shortest_path(G)
        initial_np = neighbors_preservation.compute_neig_preservation(G, all_sp=all_pairs_sp)
        print("NP:", initial_np, end=" - ")

    initial_sym = 0
    if compute_sym:
        initial_sym = ksymmetry.get_symmetric_score(G)
        print("Sym:", abs(initial_sym), end=" - ")

    initial_cr = 1
    if compute_cr:
        cr_list =crossings.count_crossings(G)
        print(cr_list, end=" - ")
        print("CR:", len(cr_list), end=" - ")
        initial_cr = max(len(cr_list), 1)

    initial_ue = 0
    if compute_ue:
        initial_ue = uniformity_edge_length.uniformity_edge_length(G)
        print("UE:", initial_ue, end=" - ")

    initial_ar = 1
    if compute_ar:
        initial_ar = areafunctions.areaerror(G)
        print("AR:", initial_ar, end=" - ")


    initial_asp = 1
    if compute_asp:
        initial_asp = areafunctions.aspectRatioerror(G)
        print("ASP:", initial_asp, end=" - ")

    print("")

    return




log = 0
def metrics_evaluator(X, print_val=False):

    '''
        Evaluates the metrics of the given layout and weights them
    '''
    global G
    global all_pairs_sp
    global log

    n = nx.number_of_nodes(G)

    #Reshape the 1D array to a n*2 matrix
    X = X.reshape((n,2))
    return_val = 0.0

    G = writeSPXPositiontoNetworkXGraph(G, X)

    ue = 0
    ue_count = 0
    if compute_ue:
        ue = uniformity_edge_length.uniformity_edge_length(G)
        ue_count = ue
        # if log%100==0:
            # print("UE:", ue, end=" - ")
        ue *= abs(compute_ue)

    st = 0
    st_count=0
    if compute_st:
        st = stress.stress(G, all_sp=all_pairs_sp)
        st_count = st
#        if log%100==0:
#        print("ST:", st, end=" - ")
        st *= abs(compute_st)/initial_st

    sym = 0
    sym_count = 0
    if compute_sym:
        G = scale_graph(G, 1000)
        sym = ksymmetry.get_symmetric_score(G)
        G = scale_graph(G, 1/1000)
        sym_count = sym
        # if log%100==0:
            # print("Sym:", abs(sym), end=" - ")
        sym = 1-sym
        sym *= abs(compute_sym)

    np = 0
    np_count = 0
    if compute_np:
        np = neighbors_preservation.compute_neig_preservation(G, all_sp=all_pairs_sp)
        np_count = np
        np = 1-np
        np *= abs(compute_np)

    cr = 0
    cr_count = 0
    if compute_cr:
        cr = len(crossings.count_crossings(G))
#        if log%100==0:
#        print("CR:", abs(cr), end=" - ")
        cr_count = cr
        cr *= abs(compute_cr)/initial_cr

    ar = 0
    ar_count = 0
    if compute_ar:
        ar = areafunctions.areaerror(G)
        ar_count = ar
        ar = abs(ar-1)
        ar *= abs(compute_ar)/initial_ar

    # Aspect ratio
    asp = 0
    asp_count = 0
    if compute_asp:
        asp = areafunctions.aspectRatioerror(G)
        asp_count = asp
        asp = abs(asp-1)
        asp *= abs(compute_asp)/initial_asp

    return_val = ue+st+sym+np+cr+ar+asp

    if print_val:
        print("score: ", return_val)

    log += 1

    return return_val

def optimize(G):


    X = netoworkxPositionsToMatrix(G)

    n = nx.number_of_nodes(G)

    # Use gradient descent to optimize the metrics_evaluator function
    # keep the X as a flattened 1D array and reshape it inside the
    # metrics_evaluator function as a 2D array/matrix
    X = X.flatten()
    res = minimize(metrics_evaluator, X, method='L-BFGS-B')

    X = res.x.reshape((n,2))


    return X


# main
# Input
#if len(sys.argv)<4:
# print('usage:python3 GD2Main.py graph_path output_folder_path metrics')
# quit()

GRAPH_PATH = sys.argv[1]
#OUTPUT_FOLDER = sys.argv[2] # Output folder
#METRICS = sys.argv[3]

input_file_name = os.path.basename(GRAPH_PATH)
# graph_name = "hi"
# #print(graph_name)
#
# G = nx.random_tree(100)

# Reading the graphs
G = nx_read_dot(GRAPH_PATH) #this should be the default structure
if not nx.is_connected(G):
    print('The graph is disconnected')
    quit()

# convert ids to integers
G = nx.convert_node_labels_to_integers(G)
G = normalize_edge_length(G)

exit()

pos = nx.planar_layout(G)
print(pos)

## Set zero coordinates for all vertices
for i in pos.keys():
    x = round(pos[i][0], 8)
    y = round(pos[i][1],8)
    curr_pos = str(x)+","+str(y)
    nx.set_node_attributes(G, {i:curr_pos}, "pos")

print(nx.get_node_attributes(G, "pos"))


print(nx.info(G))

# Metrics weights
compute_ue=0 #Uniformity Edge lengths
compute_st=0 # Stress
compute_sym=0 # Symmetry
compute_np=0 # Neighbor Preservation
compute_cr=0 #Crossings
compute_ar=0 #Area
compute_asp=0 #Aspect ratio

#if METRICS=='0':
#    compute_ue = 1
#elif METRICS=='1':
#    compute_st = 1
#elif METRICS=='2':
#    compute_sym = 1
#elif METRICS=='3':
#    compute_np = 1
#elif METRICS=='4':
#    compute_cr = 1
#elif METRICS=='5':
#    compute_ar = 1
#elif METRICS=='6':
#    compute_asp = 1

#compute_st = 1


# Metric specific global variables
initial_st = 1
initial_cr = 1
initial_ar = 1
initial_asp = 1
all_pairs_sp = None

G = normalize_edge_length(G)
print("Initial metrics")
write_dot(G, 'output/' + graph_name + '_initial.dot')
printMetrics(G)
final_position_matrix = optimize(G)
G = writeSPXPositiontoNetworkXGraph(G, final_position_matrix)
G = normalize_edge_length(G)
write_dot(G, 'output/' + graph_name + '_final.dot')
print("Final Metrics")
printMetrics(G)
metrics_evaluator(final_position_matrix, print_val=True)
