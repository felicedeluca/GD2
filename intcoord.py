import networkx as nx
import math

def overlapping(G):

    pos = nx.get_node_attributes(G, "pos")
    overlapping  = 0
    for v in nx.nodes(G):
        for u in nx.nodes(G):

            if u==v:
                continue

            x_u = round(float(pos[u].split(",")[0]),5)
            x_v = round(float(pos[v].split(",")[0]),5)
            y_u = round(float(pos[u].split(",")[1]),5)
            y_v = round(float(pos[v].split(",")[1]),5)

            if (x_u == x_v and y_u == y_v):
                overlapping += 0.5


    return overlapping



def x_span(G, desired_span=1):
    '''
        The desired x-span between two vertices on the same line of a grid
        should be 1.
        if the value is lower then it is a penanlty.
    '''

    pos = nx.get_node_attributes(G, "pos")

    x_span = 0

    epsilon_for_same_y = 0.0000000001
    for v in nx.nodes(G):
        for u in nx.nodes(G):
            x_u = round(float(pos[u].split(",")[0]),5)
            x_v = round(float(pos[v].split(",")[0]),5)
            y_u = round(float(pos[u].split(",")[1]),5)
            y_v = round(float(pos[v].split(",")[1]),5)

            if y_u != y_v:
                continue

            if abs(x_v-x_u)<desired_span:
                x_span+=abs(x_v-x_u)

    return x_span

def upwardness(G):
    '''
    Evaluates the distance between two ajacent vertices needed to get upaward edge
    '''

    pos = nx.get_node_attributes(G, "pos")

    if not nx.is_directed(G):
        return 0

    non_upward = 0

    epsilon_for_same_y = 0.0000000001

    for e in nx.edges(G):

        (u, v) = e

        y_u = float(pos[u].split(",")[1])
        y_v = float(pos[v].split(",")[1])

        if y_u < (y_v-1):
            continue

        non_upward += epsilon_for_same_y + (y_u-y_v)+1

    return non_upward



def nonintvalues(G):

    pos = nx.get_node_attributes(G, "pos")

    nonint = 0

    for v in pos.keys():

        x = float(pos[v].split(",")[0])
        y = float(pos[v].split(",")[1])


        x_number_dec = float(abs(x-round(x,0)))
        y_number_dec = float(abs(y-round(y,0)))

        # print(x, y, x_number_dec, y_number_dec)

        nonint += x_number_dec
        nonint += y_number_dec

    return nonint

def upwardgrid(G):

    score = 0
    score += nonintvalues(G)
    score += upwardness(G)
    score += x_span(G)
    score += overlapping(G)

    return score
