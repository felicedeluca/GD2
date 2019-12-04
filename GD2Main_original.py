# Utils
import math
import sys
import time
import os

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

def runOptimizer(G):
    '''
        Set inital values before optimization
    '''

    global initial_st
    global all_pairs_sp
    global initial_cr
    # Do some preliminary stuff

    # Stress will be normalized considering the first value as max
    # To speed up ST precompute all pairs shortest paths
    initial_st = 1
    if compute_st:
        initial_st = stress.stress(G, all_sp=all_pairs_sp)
        print("Initial ST:", initial_st, end=" - ")
        if all_pairs_sp is None:
            all_pairs_sp = nx.shortest_path(G)

    # To speed up NP precompute all pairs shortest paths
    if compute_np:
        if all_pairs_sp is None:
            all_pairs_sp = nx.shortest_path(G)


    # Crossings steup
    initial_cr = 1
    if compute_cr:
        initial_cr = len(crossings.count_crossings(G))


    # Save the values to a file.

    return optimize(G)


log = 0

def metrics_evaluator(X):
    '''
        Evaluates the metrics of the given layout and weights them
    '''
    global log
    global G
    global all_pairs_sp

    n = nx.number_of_nodes(G)

    #Reshape the 1D array to a n*2 matrix
    X = X.reshape((n,2))
    return_val = 0.0

    G = writeSPXPositiontoNetworkXGraph(G, X)

    ue = 0
    if compute_ue:
        ue = uniformity_edge_length.uniformity_edge_length(G)
        if log%100==0:
            print("UE:", ue, end=" - ")
        ue *= abs(compute_ue)

    st = 0
    if compute_st:
        st = stress.stress(G, all_sp=all_pairs_sp)
        if log%100==0:
            print("ST:", st, end=" - ")
        st *= abs(compute_st)/initial_st

    sym = 0
    if compute_sym:
        sym = ksymmetry.get_symmetric_score(G)
        if log%100==0:
            print("Sym:", abs(sym), end=" - ")
        sym = 1-sym
        sym *= abs(compute_sym)

    np = 0
    if compute_np:
        np = neighbors_preservation.compute_neig_preservation(G, all_sp=all_pairs_sp)
        if log%100==0:
            print("NP:", abs(np), end=" - ")
        np = 1-np
        np *= abs(compute_np)

    cr = 0
    if compute_cr:
        cr = len(crossings.count_crossings(G))
        if log%100==0:
            print("cr", cr, end="-")
        cr *= abs(compute_cr)/initial_cr


    return_val = ue+st+sym+np+cr
    if log%100==0:
        print("ret", return_val)
        write_dot(G, 'output/' + graph_name + '_running.dot')

    log +=1


    return return_val

def optimize(G):


    X_curr = netoworkxPositionsToMatrix(G)
    X = np.copy(X_curr)
    X_prev = np.copy(X_curr)

    n = nx.number_of_nodes(G)

    num_iters = 0

    eps_step = 0.01
    g_tol = 0.1

    while 1:
        num_iters += 1

        X = np.copy(X_curr)

        # Use gradient descent to optimize the metrics_evaluator function
        # keep the X as a flattened 1D array and reshape it inside the
        # metrics_evaluator function as a 2D array/matrix
        X = X.flatten()
        res = minimize(metrics_evaluator, X, method='BFGS', options={'gtol': g_tol, 'disp': True, 'eps':eps_step})

        X_prev = np.copy(X_curr)
        X_curr = res.x.reshape((n,2))

        if IS_LOG_COST_FUNCTION:
            COST_FUNCTIONS[num_iters-1] = metrics_evaluator(res.x)

        X = np.copy(X_curr)

        print("Iteration", num_iters, end = "--")
        print("eps_step", eps_step, end = "--")
        print("g_tol", g_tol)

        eps_step/=10
        g_tol/=10

        # # Termination conditions
        # if (not USE_NUM_ITERS):
        #     print("converged")
        #     prev_total_score = metrics_evaluator(X_prev)
        #     curr_total_score = metrics_evaluator(X_curr)
        #     score_improvement = (prev_total_score - curr_total_score) / prev_total_score
        #     print("Score improvement:",score_improvement)
        #     if(score_improvement < EPSILON):
        #         return X_curr
        # else:
        if(num_iters >= NUM_ITERATIONS):
            print("Max iterations reached", num_iters, NUM_ITERATIONS)
            return X_curr


# main
# Input
if len(sys.argv)<4:
 print('usage:python3 GD2Main.py graph_path output_folder_path metrics')
 quit()

GRAPH_PATH = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2] # Output folder
METRICS = sys.argv[3]

input_file_name = os.path.basename(GRAPH_PATH)
graph_name = input_file_name.split(".")[0]
print(graph_name)

# Reading the graphs
G = nx_read_dot(GRAPH_PATH) #this should be the default structure
if not nx.is_connected(G):
    print('The graph is disconnected')
    quit()

# convert ids to integers
G = nx.convert_node_labels_to_integers(G)


# Metrics weights
compute_ue=0 #Uniformity Edge lengths
compute_st=1 # Stress
compute_sym=0 # Symmetry
compute_np=0 # Neighbor Preservation
compute_cr=1 #Crossings

# Gradient Descente convergence threshold
EPSILON = 0.000001
USE_NUM_ITERS = True
IS_LOG_COST_FUNCTION = True
NUM_ITERATIONS = 30 # Number of Gradient Descent iterations
COST_FUNCTIONS = -np.ones((NUM_ITERATIONS));


# Metric specific global variables
initial_st = 1
initial_cr = 1
all_pairs_sp = None


G = scale_graph(G, 100)
# run the optimizer
final_position_matrix = runOptimizer(G)
G = writeSPXPositiontoNetworkXGraph(G, final_position_matrix)

# Write the graph in a dot file
write_dot(G, 'output/' + graph_name + '_final.dot')
