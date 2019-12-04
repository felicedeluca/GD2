import networkx as nx
import sys
import os
from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_agraph import read_dot as nx_read_dot

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
