import networkx as nx
import numpy as np
import itertools
from scipy.spatial import distance
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
import pdb
import globals as const
from funcs_orig import *
from math import isclose
from pyqt_viz import edge_viewer
import time

# Constants for simulation
dt = const.dt
#var_dt = True

# dimensions of the cell 
l_apical = const.l_apical 
l_depth = const.l_depth 

# Set the arcs list
inner_arc = const.inner_arc
outer_arc = const.outer_arc

# mechanical parameters
# rest lengths of the passively elastic apical, basal and cell walls
l0_apical = l_apical
l0_basal = l_apical 
l0_wall = l_depth 

mu_apical = const.mu_apical         
mu_basal = const.mu_basal          
mu_wall = const.mu_wall          
myo_beta = const.myo_beta 
eta = const.eta 
press_alpha = const.press_alpha 
l_mvmt = const.l_mvmt

# initialize the tissue
G, K, centers, num_api_nodes, circum_sorted, belt, triangles = tissue_3d()
pit_centers = const.pit_centers 

# Starting from t=0
t = 0
num_inter = 0 
blacklist = [] 
contract = [True for counter in range(0,num_inter)]
#inter_edges = [[301,302],[295,296],[292,293],[298,299],[45,46],[39,40],[272,273],[174,175],[180,181],[276,277],[183,184],[177,178],[112,113],[286,287],[289,290],[115,116],[109,110],[283,284],[280,281],[106,107]] 
# Starting from t=? after intercalations occur
#t = 1640 
#num_inter = 20     
#blacklist = [[301,302],[295,296],[292,293],[298,299],[45,46],[39,40],[272,273],[174,175],[180,181],[276,277],[183,184],[177,178],[112,113],[286,287],[289,290],[115,116],[109,110],[283,284],[280,281],[106,107]] 
#contract = [False for counter in range(0,num_inter)]
#G = nx.read_gpickle('/home/cdurney/3d-vertex/concentric/t1640.pickle')
#
#for counter in range(0,num_inter): 
#    node = blacklist[counter][0]
#    neighbor = blacklist[counter][1]
#    print(node, neighbor) 
#    cents = list(set(K.neighbors(node)) & set(K.neighbors(neighbor)))
#    ii = list((set(list(K.neighbors(node))) & set(list(centers))) - (set(list(K.neighbors(node))) & set(list(K.neighbors(neighbor)))))[0]
#    jj = list((set(list(K.neighbors(neighbor))) & set(list(centers))) - (set(list(K.neighbors(node))) & set(list(K.neighbors(neighbor)))))[0]
#    temp1 = list(set(K.neighbors(node)) & set(K.neighbors(cents[0])))
#    temp1.remove(neighbor)
#    temp2 = list(set(K.neighbors(neighbor)) & set(K.neighbors(cents[1])))
#    temp2.remove(node)
#    circum_sorted, triangles, K = new_topology(K,[node, neighbor], cents, temp1, temp2, ii, jj, belt, centers, num_api_nodes)
#

# t=initial nx Graph in pickled form for plotting later
print(t) 
file_name = 't' + str(int(t)) 
nx.write_gpickle(G,file_name + '.pickle')
np.save(file_name,circum_sorted) 

viewer = edge_viewer(G,attr='myosin')

t_plot=5
t_last=-t_plot
while t <= const.t_final:
    
    if t == const.t_1:
        for i in range(0,len(inner_arc)):
            G[inner_arc[i-1]][inner_arc[i]]['myosin'] = const.belt_strength     
        print("Inner arc established")

    # update myosin on outer arc 
    if t == const.t_2:
        for i in range(0,len(outer_arc)):
            G[outer_arc[i-1]][outer_arc[i]]['myosin'] = const.belt_strength     
        print("Outer arc established")

    # update myosin on belt
    if t == const.t_belt:
        for i in range(0,len(belt)):
            G[belt[i-1]][belt[i]]['myosin'] = const.belt_strength     
        print("Belt established") 

    if t-t_last>=t_plot:
        viewer(G)
    # increment t by dt
    # initialize force_dict back to zeros
    t = round(t+dt,1)
    print(dt, t) 
    pos = nx.get_node_attributes(G,'pos')
    force_dict = {new_list: np.zeros(3,dtype=float) for new_list in G.nodes()} 
    
    # pre-calculate magnitude of pressure
    # index of list corresponds to index of centers list
    PI = np.zeros(len(centers),dtype=float) 
    # eventually move to classes?
    for n in range(0,len(centers)):
        # get nodes for volume
        pts = get_points(G,centers[n],pos) 
        # calculate volume
        vol = convex_hull_volume_bis(pts)  
        # calculate pressure
        PI[n] = -press_alpha*(vol-const.v_0) 

#    # Update myosin on a fictitious pit (no resemblance to SG geometry)
#    if t < const.t_pit: 
#        myo = const.pit_strength*t
#        for node in pit_centers: 
#            if node == 0:
#                myo = 1.5*myo
#            for neighbor in G.neighbors(node): 
#                G[node][neighbor]['myosin'] = myo

#    if t > const.t_intercalate:
#        if contract[0] == True:
#            G[301][302]['myosin'] = const.belt_strength*(t-const.t_intercalate) 
    
    # update myosin on inner arc 
   

    for node in G.nodes(): 
        # update force on each node  
        force = [0.0,0.0,0.0]
    
        # Elastic forces due to the cytoskeleton 
        for neighbor in G.neighbors(node):
            a = pos[node]
            b = pos[neighbor]
            
            dist = distance.euclidean(a,b)
            direction = unit_vector(a,b)
            
            magnitude = elastic_force(dist, G[node][neighbor]['l_rest'], mu_apical) 
            force = np.sum([force,magnitude*np.array(direction)],axis=0)
            
            # Force due to myosin
            magnitude = myo_beta*G[node][neighbor]['myosin']
            force = np.sum([force, magnitude*np.array(direction)],axis=0)

        force_dict[node] = np.add(force_dict[node], force) 
    
    for center in centers:
        index = centers.index(center)
        pts = circum_sorted[index]
        centroid = np.array([pos[center], pos[center+1000]])
        centroid = np.average(centroid,axis=0)
        
        # pressure for: 
        # apical nodes     
        for i in range(0,len(circum_sorted[index])):
            area, extra = be_area([center,pts[i],pts[i-1]],[center,pts[i],pts[i-1]],pos) 
            magnitude = PI[index]*area[0]*(1/3)
            
            direction = area[1]/np.linalg.norm(area[1]) 
            force = magnitude*direction
            force_dict[center] = np.add(force_dict[center],force)
            force_dict[pts[i-1]] = np.add(force_dict[pts[i-1]],force)
            force_dict[pts[i]] = np.add(force_dict[pts[i]],force)
   
        # pressure for: 
        # basal nodes
            area, extra = be_area([center+1000,pts[i-1]+1000,pts[i]+1000],[center+1000,pts[i-1]+1000,pts[i]+1000],pos) 
            magnitude = PI[index]*area[0]*(1/3)
            direction = area[1]/np.linalg.norm(area[1]) 
            force = magnitude*direction
            force_dict[center+1000] = np.add(force_dict[center+1000],force)
            force_dict[pts[i-1]+1000] = np.add(force_dict[pts[i-1]+1000],force)
            force_dict[pts[i]+1000] = np.add(force_dict[pts[i]+1000],force)

    # pressure for side panels
    # loop through each cell
    for index in range(0,len(circum_sorted)):
        cell_nodes = circum_sorted[index]
        centroid = np.array([pos[centers[index]], pos[centers[index]+1000]])
        centroid = np.average(centroid, axis=0)
        # loop through the 6 faces (or 5 or 7 after intercalation)
        for i in range(0, len(cell_nodes)):
            pts_id = np.array([cell_nodes[i-1], cell_nodes[i], cell_nodes[i]+1000, cell_nodes[i-1]+1000])
            pts_pos = np.array([pos[pts_id[ii]] for ii in range(0,4)])
            # on each face, calculate the center
            center = np.average(pts_pos,axis=0)
            # loop through the 4 triangles that make the face
            for ii in range(0,4):
                pos_side = [center, pts_pos[ii-1], pts_pos[ii]] 
                area = area_side(pos_side) 
                magnitude = PI[index]*area[0]*(1/2)
                
                direction = area[1]/np.linalg.norm(area[1]) 
                force = magnitude*direction
                force_dict[pts_id[ii-1]] = np.add(force_dict[pts_id[ii-1]],force)
                force_dict[pts_id[ii]] = np.add(force_dict[pts_id[ii]],force)
    
    # Implement bending energy
    # Loop through all alpha, beta pairs of triangles
    for pair in triangles:
        alpha, beta = pair[0], pair[1]
        
        # Apical faces, calculate areas and cross-products 
        A_alpha, A_beta = be_area(alpha, beta, pos)
         
        for node in alpha:
            inda = alpha.index(node) 
            nbhrs_alpha = (alpha[(inda+1)%3], alpha[(inda-1)%3]) 
            if node in beta:
                indb = beta.index(node) 
                nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
                frce = const.c_ab*bending_energy(nbhrs_alpha, nbhrs_beta, A_alpha, A_beta, pos)
            else:
                frce = const.c_ab*bending_energy(nbhrs_alpha, False, A_alpha, A_beta, pos)
		
            force_dict[node] = np.add(force_dict[node],frce)

        for node in beta:
            # don't double count the shared nodes
            indb = beta.index(node) 
            nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
            if node not in alpha:
                frce = const.c_ab*bending_energy(False, nbhrs_beta, A_alpha, A_beta, pos)
            else:			
                frce = const.c_ab*np.array([0.,0.,0.])
            
            force_dict[node] = np.add(force_dict[node],frce)

        # Basal faces
        alpha = [alpha[0]+1000, alpha[1]+1000, alpha[2]+1000] 
        beta = [beta[0]+1000, beta[1]+1000, beta[2]+1000] 

        A_alpha, A_beta = be_area(alpha, beta, pos)
        
        for node in alpha:
            inda = alpha.index(node) 
            nbhrs_alpha = (alpha[(inda+1)%3], alpha[(inda-1)%3]) 
            if node in beta:
                indb = beta.index(node) 
                nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
                frce = const.c_ab*bending_energy(nbhrs_alpha, nbhrs_beta, A_alpha, A_beta, pos)
            else:
                frce = const.c_ab*bending_energy(nbhrs_alpha, False, A_alpha, A_beta, pos)
		
            force_dict[node] = np.add(force_dict[node],frce)

        for node in beta:
            # don't double count the shared nodes
            indb = beta.index(node) 
            nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
            if node not in alpha:
                frce = const.c_ab*bending_energy(False, nbhrs_beta, A_alpha, A_beta, pos)
            else:			
                frce = np.array([0.,0.,0.])
            
            force_dict[node] = np.add(force_dict[node],frce)

    # update location of node 
    pos = nx.get_node_attributes(G,'pos')
    
    for node in force_dict:
        G.node[node]['pos'] = d_pos(pos[node],force_dict[node],dt)

   ## Check for intercalation events
    pos = nx.get_node_attributes(G,'pos')
    for node in range(0,num_api_nodes):
        if node not in belt: 
            for neighbor in G.neighbors(node):
                if (neighbor < 1000) and (neighbor not in belt) and (node not in centers) and (neighbor not in centers) and ([min(node, neighbor), max(node, neighbor)] not in blacklist): 
                 
                    a = pos[node]
                    b = pos[neighbor]
                    c = pos[node+1000]
                    d = pos[neighbor+1000]
                    
                    dist = distance.euclidean(a,b)
                    
                    if (dist < const.l_intercalation): 
                        if (np.random.rand(1)[0] < 1.):
                            print("Intercalation event between nodes", node, "and", neighbor, "at t = ", t) 
                            # collapse nodes to same position 
                            # apical  
                            avg_loc = (np.array(a) + np.array(b)) / 2.0 
                            a = avg_loc 
                            b = avg_loc 
                            # basal 
                            avg_loc = (np.array(c) + np.array(d)) / 2.0 
                            c = avg_loc 
                            d = avg_loc 
                            # move nodes toward new center 
                            # apical 
                            cents = list(set(G.neighbors(node)) & set(G.neighbors(neighbor)))
                            mvmt = unit_vector(a,pos[cents[1]])
                            a = [a[0]+l_mvmt*mvmt[0], a[1]+l_mvmt*mvmt[1], a[2]+l_mvmt*mvmt[2]]
                            G.node[node]['pos'] = a 
                            mvmt = unit_vector(b,pos[cents[0]])
                            b = [b[0]+l_mvmt*mvmt[0], b[1]+l_mvmt*mvmt[1], b[2]+l_mvmt*mvmt[2]]
                            G.node[neighbor]['pos'] = b 
                            # basal 
                            #cents = list(set(G.neighbors(node+1000)) & set(G.neighbors(neighbor+1000)))
                            mvmt = unit_vector(c,pos[cents[1]+1000])
                            c = [c[0]+l_mvmt*mvmt[0], c[1]+l_mvmt*mvmt[1], c[2]+l_mvmt*mvmt[2]]
                            G.node[node+1000]['pos'] = c 
                            mvmt = unit_vector(d,pos[cents[0]+1000])
                            d = [d[0]+l_mvmt*mvmt[0], d[1]+l_mvmt*mvmt[1], d[2]+l_mvmt*mvmt[2]]
                            G.node[neighbor+1000]['pos'] = d 
                            
                            ii = list((set(list(G.neighbors(node))) & set(list(centers))) - (set(list(G.neighbors(node))) & set(list(G.neighbors(neighbor)))))[0]
                            jj = list((set(list(G.neighbors(neighbor))) & set(list(centers))) - (set(list(G.neighbors(node))) & set(list(G.neighbors(neighbor)))))[0]
                            temp1 = list(set(G.neighbors(node)) & set(G.neighbors(cents[0])))
                            temp1.remove(neighbor)
                            temp2 = list(set(G.neighbors(neighbor)) & set(G.neighbors(cents[1])))
                            temp2.remove(node)

                            # sever connections
                            # apical   
                            G.remove_edge(node,cents[0])
                            G.remove_edge(node,temp1[0])
                            G.remove_edge(neighbor,cents[1])
                            G.remove_edge(neighbor,temp2[0])
                            # basal 
                            G.remove_edge(node+1000,cents[0]+1000)
                            G.remove_edge(node+1000,temp1[0]+1000)
                            G.remove_edge(neighbor+1000,cents[1]+1000)
                            G.remove_edge(neighbor+1000,temp2[0]+1000)

                            # add new connections
                            # apical 
                            # new edges 
                            G.add_edge(node,temp2[0],l_rest = const.l_apical, myosin=0,color='#808080')
                            G.add_edge(neighbor,temp1[0],l_rest = const.l_apical, myosin=0,color='#808080')
                            # new spokes 
                            G.add_edge(neighbor,ii,l_rest = const.l_apical, myosin=0)
                            G.add_edge(node,jj,l_rest = const.l_apical, myosin=0)
                            # basal 
                            # new edges 
                            G.add_edge(node+1000,temp2[0]+1000,l_rest = const.l_apical, myosin=0,color='#808080')
                            G.add_edge(neighbor+1000,temp1[0]+1000,l_rest = const.l_apical, myosin=0,color='#808080')
                            # new spokes 
                            G.add_edge(neighbor+1000,ii+1000,l_rest = const.l_apical, myosin=0)
                            G.add_edge(node+1000,jj+1000,l_rest = const.l_apical, myosin=0)
                            
                            # reset myosin on contracted edge
                            G[node][neighbor]['myosin'] = 0
                            G[node+1000][neighbor+1000]['myosin'] = 0
                            
                            blacklist.append([min(node, neighbor), max(node, neighbor)])
                            
                            circum_sorted, triangles, K = new_topology(K,[node, neighbor], cents, temp1, temp2, ii, jj, belt, centers, num_api_nodes)
                            
                            if min(node,neighbor) == 301:
                                contract[0] = False

#    #set dt for next loop 
#    if var_dt == True:
#        if any(contract) == True:
#            # if any edges are still contracting, check for threshold length 
#            for i in range(0,num_inter):
#            # calculate lengths of those that are still True 
#                if contract[i] == True:
#                    a = inter_edges[i][0]
#                    b = inter_edges[i][1]
#                    if distance.euclidean(pos[a],pos[b]) < 0.2:
#                        dt = 0.1
#                        break 
#        else: 
#            if isclose(t % 1, 0) == False:       
#                dt = 0.1 
#            else:
#                dt = const.dt
#                var_dt = False 
#    else:
#        dt  = const.dt

# Save nx Graph in pickled form for plotting later
    
    if t % 1 == 0: 
        file_name = 't' + str(round(t)) 
        nx.write_gpickle(G,file_name + '.pickle')
        np.save(file_name,circum_sorted)
