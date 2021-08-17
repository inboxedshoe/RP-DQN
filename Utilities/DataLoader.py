#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import networkx as nx
import matplotlib.pyplot as plt
import dgl


def PrepareGraphs():
    
    folder = 'Data_TSP'                     #location of the data wrt to the script
    fname = ('%s/tsp2d.tsp' % (folder))     #data file name
    coors = {}
    in_sec = False
    n_nodes = -1
    
    #this part is loopable for multiple files
    with open(fname, 'r') as f_tsp:
        for l in f_tsp:
            if 'DIMENSION' in l:            #get number nodes to create
                n_nodes = int(l.split(' ')[-1].strip())
            if in_sec:                      #save coordinates to list
                idx, x, y = [int(w.strip()) for w in l.split(' ')]
                coors[idx - 1] = [float(x) / 1000000.0, float(y) / 1000000.0]
                assert len(coors) == idx
            elif 'NODE_COORD_SECTION' in l:
                in_sec = True
    assert len(coors) == n_nodes            #check no' coordinates and no' nodes are the same
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))        #add coordinates as features
    nx.set_node_attributes(g, name = 'pos', values = coors)
    
    g = dgl.DGLGraph(g)    #convert networkx graph to dgl
    nx.draw(g.to_networkx(), with_labels=True)
    plt.show()             #visualize graph
    
    return(g)
    
g = PrepareGraphs()
