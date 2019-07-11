#START of MITOCHONDRIA ANALYSIS
import sys
import re
import json
import csv
import gzip
import logging
from io import BytesIO
from functools import partial
from multiprocessing.pool import ThreadPool
from collections import defaultdict
import numpy as np
import pandas as pd
#START of MITOCHONDRIA ANALYSIS
import re
import json
import csv
import gzip
import logging
from io import BytesIO
from functools import partial
from multiprocessing.pool import ThreadPool
from collections import defaultdict
import numpy as np
import pandas as pd
from requests import HTTPError
from libdvid import DVIDNodeService, encode_label_block
from neuclease.dvid.server import fetch_server_info
import neuclease
from neuclease.dvid.labelmap import fetch_labelmap_voxels
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import skimage
from skimage import external
from scipy.spatial.distance import euclidean
from skimage.graph import MCP_Geometric
import numpy as np
#from bokeh.plotting import figure, show, output_file
#from bokeh.io import output_notebook
#output_notebook()
from neuclease.dvid import fetch_labelmap_voxels
from bokeh.palettes import Category20
from dvidutils import LabelMapper
import neuprint as neu
import pandas as pd
from pandas import DataFrame
from neuprint import Client
from tqdm import tqdm_notebook
from numpy import size
from numpy import dtype, array
#import neuroglancer
#neuroglancer.set_server_bind_address('0.0.0.0')
from numpy import transpose
from neuclease.dvid import *
#from sklearn.cluster import KMeans


def main():
    neuron_id = sys.argv[1]
    result = process_neuron_mito(neuron_id)
    if result is not True:
        print("Oh crap, something went wrong")
        sys.exit(1)
    
    print("DONE.")

# we preprocess to gather all presynapses of given ID
def preprocess(body_ID):
    body_ID = body_ID
    Neurons_FB = pd.read_csv('Neurons_FB.csv')
    y = Neurons_FB[Neurons_FB.bodyId == int(body_ID)]
    y = y[y.type == 'pre']
    return list(y[['x','y','z']].values), len(list(y[['x','y','z']].values))


#Helper Functions before main one
def is_mito(mito, vol):
    '''
    Pass in the binary image sets and overlap
    Tests if mitochondria can be found in the neuron of certain fetched box
    '''
    test_mito = np.where(mito == 4, 0, 1)
    #test_vol2 = np.copy(vol)
    #test_vol2[test_vol2 != 1] = 0
    #test_vol2 = np.transpose(test_vol2.nonzero())
    #test_vol2 = list(map(tuple, test_vol2))
    #neuron_set = set(test_vol2)
    #mito = list(map(tuple, mito))
    #mito_set = set(mito)
    test_mito = test_mito.astype('uint64')
    #print(test_mito.shape)
    #print(transpose(test_mito.nonzero()),)
    #print(np.sum(np.array(test_mito) & np.array(vol)))
    if np.sum(np.array(test_mito) & np.array(vol)) > 0:
        return True
    else: 
        return False
    
def is_synapse_neuron(vol, syn_cords):
    '''
    Tests if the synapse is in the neuron in terms of segmentation
    '''
    neuron_set = transpose(vol.nonzero())#np.where(vol == BodyID, 1, 0)
    neuron_set = [tuple(a) for a in neuron_set]
    syn_cords = [tuple(b) for b in syn_cords]
    #synapse_set = set([(x,y,z) for x in range(99,100) for y in range(99,100) for z in range(99,100)])
    #vol = np.copy(neuron)
    #vol[vol != 1] = 0
    #vol = np.transpose(neuron.nonzero())
    #vol = list(map(tuple, vol))
    #neuron_set = set(vol)
    if syn_cords[0] in neuron_set:
         return True
    else:
        return False

def recoordinate_synapses(vol, syn_cords, r):
    '''
    Reorients the synapses towards its nearest valid neighbors and returns the new coordinates as a group of starting coordinates
    
    Parameters: 
                vol = fetched volume from DVID
                syn_cords = list of lists containing only local coordinates of synapse
                r = size of range of new coordinates that we represent as the new synapse: ex. r = 1 means we look only at the neighboring voxels of the synapse
    '''
    x, y, z = syn_cords[0][0], syn_cords[0][1], syn_cords[0][2]
    np.array(vol)
    search_vol = vol[x-r:x+r+1,y-r:y+r+1,z-r:z+r+1]
    search_cords = np.transpose(search_vol.nonzero())
    real_cords = search_cords + np.array([x,y,z])
    return real_cords



def boundary_box(seg_mask):
    seg_mask = np.where(seg_mask == True, 1, 0)
    raw_cords = np.transpose(seg_mask.nonzero())
    return seg_mask, (np.amin(raw_cords, axis = 0), np.amax(raw_cords, axis = 0))

def remove_empty(dicts):
    new_dicts = {}
    lsts = [*dicts]
    for i in lsts:
        new_dicts[i] = list(filter(None, dicts[i]))
    return new_dicts


def Mito_segregation(Mito_Dists, Body_ID):
    MITO1_DISTS = []
    MITO2_DISTS = []
    MITO3_DISTS = []
    Mito_Dists = remove_empty({Body_ID:Mito_Dists[int(Body_ID)]})
    for i in [*Mito_Dists]:
        for j in range(len(Mito_Dists[i])):
            if Mito_Dists[i][j][0][1] == 1:
                MITO1_DISTS.append(Mito_Dists[i][j][0][0])
            elif Mito_Dists[i][j][0][1] == 2:
                MITO2_DISTS.append(Mito_Dists[i][j][0][0])
            else:
                MITO3_DISTS.append(Mito_Dists[i][j][0][0])
    return MITO1_DISTS, MITO2_DISTS, MITO3_DISTS

def Mito_Synapse_Distance(local_syn, seg_vol, seg_mito, Synapse):    
    
    #First we got to initialize some empty lists and constants that we will store the list of distributions in
    i = 8
    dist_lst = []
    errors_from_no_mito = []
    errors_from_synapse_loc = []
    errors_from_scale = []
    #First we need to fetch the Neuron synapse info from neuprint
   
    #for i in range(len(synapses)):
    #    Synapse = list(synapses[i])

        #We define the centerpoint of our data we fetch as the synapse location
        #We define the dimensions of the box as well as fetch the segmentation in vol and mitochondria data as mito
        #We then manipulate vol to make any neurons that is not the desired neuron have a cost of infinity, and the desired neuron
        #a cost of 1

    #if scale == 0:
    #    i = 1
    #else:
    #scale = 3
    #i = 2**scale

    #Synapse = synapse_cords[0]
    seg_vol = seg_vol.astype('uint64')
    subvol = seg_vol.astype('float64')
    #print([31, 31, 14] in np.transpose(subvol.nonzero()))
    #print(subvol[31][31][14])
    subvol[subvol == 0] = np.inf
    #print(subvol[31][31][14])
    #print(subvol)
    #subvol[subvol != Body_ID] = float('inf')
    #subvol[subvol == Body_ID] = 1
    #subvol = np.where(subvol == Body_ID, 1, float('inf'))
    #Next we manipulate the mitochondria data, such that all mitochondria will be indexed if they have a cost of 1
    #Therefore, we set all voxels that were not mitochondria to 0 and took each coordinate for type of mito, to then stack together
    #This will serve as our end targets when we run the cost path

    #Mito1 = np.copy(mito)
    #Mito2 = np.copy(mito)
    #Mito3 = np.copy(mito)
    #Mito1[Mito1 != 1] = 0
    #Mito1 = np.transpose(Mito1.nonzero())
    #Mito2[Mito2 != 2] = 0
    #Mito2 = np.transpose(Mito2.nonzero())
    #Mito3[Mito3 != 3] = 0
    #Mito3 = np.transpose(Mito3.nonzero())
    #Mito_coordinates = np.vstack((Mito1,Mito2,Mito3))
    Mito_adjust = np.where(seg_mito == 4, 0, 1)
    Mito_coordinates = np.transpose(Mito_adjust.nonzero())
    #print(is_mito(seg_mito, seg_vol))
    #print(is_synapse_neuron(seg_vol, local_syn))
    if is_mito(seg_mito, seg_vol) == True:
        if is_synapse_neuron(seg_vol, local_syn) == True:
            #This is the object created solely for the cost path, in order to find the relative distance between synapse and mito
            #Note that we found the minimum cost for the path because that is considered the closest distance between synapse and mito
            #We also set the cost function to find the first end point, which is technically the shortest distance we are looking for

            mcp = MCP_Geometric(subvol)
            cum_costs, tb = mcp.find_costs(local_syn, Mito_coordinates, find_all_ends = False)
            #target_dist = np.amin(cum_costs[tuple(Mito_coordinates.transpose())])
            mito_indexes = tuple(Mito_coordinates.transpose())
            ii = cum_costs[mito_indexes].argmin()
            target_distance = cum_costs[mito_indexes][ii]
                
            mito_voxels = seg_mito[mito_indexes]
            min_dist_mito_type = mito_voxels[ii]
            min_dist_zyx = Mito_coordinates[ii]
            #print(min_dist_mito_type)
            if target_distance == 0:
                dist_lst.append((i*1.0, min_dist_mito_type))
            elif i*target_distance == np.inf:
                errors_from_scale.append(Synapse)
            else:
                assert i*target_distance != np.inf
                dist_lst.append((i*target_distance, min_dist_mito_type))
        else:
            new_cords = recoordinate_synapses(seg_vol, local_syn, 1)
            mito_indexes = tuple(Mito_coordinates.transpose())
            if len(new_cords) == 0:
                new_cords = recoordinate_synapses(seg_vol, local_syn, 2)
                if len(new_cords) == 0:
                    errors_from_synapse_loc.append(Synapse)
                else:
                    mcp = MCP_Geometric(subvol)
                    cum_costs, tb = mcp.find_costs(new_cords, Mito_coordinates, find_all_ends = False)
                    #new_target_dist = np.amin(cum_costs[tuple(Mito_coordinates.transpose())])
                    mito_indexes = tuple(Mito_coordinates.transpose())
                    ii = cum_costs[mito_indexes].argmin()
                    new_target_distance = cum_costs[mito_indexes][ii]
                
                    mito_voxels = seg_mito[mito_indexes]
                    min_dist_mito_type = mito_voxels[ii]
                    min_dist_zyx = Mito_coordinates[ii]
                    dist_lst.append((i*new_target_distance, min_dist_mito_type))
                    
                    if i*new_target_distance == np.inf:
                        errors_from_scale.append(Synapse)
                    else:
                        assert i*new_target_distance != np.inf
                        dist_lst.append((i*new_target_distance, min_dist_mito_type))
            else:
                mcp = MCP_Geometric(subvol)
                cum_costs, tb = mcp.find_costs(new_cords, Mito_coordinates, find_all_ends = False)

                #cum_costs_mito_only = np.empty_like(cum_costs)
                #cum_costs_mito_only[:] = np.inf
                #cum_costs_mito_only[tuple(Mito_coordinates.transpose())] = cum_costs[tuple(Mito_coordinates.transpose())]

                #i = np.unravel_index(cum_costs_mito_only.argmin(), cum_costs_mito_only.shape)
                #min_distance = cum_costs_mito_only[i]
                #min_distance_mito_type = seg_mito[i]
                #print(tuple(Mito_coordinates.transpose()))

                ii = cum_costs[mito_indexes].argmin()
                new_target_distance = cum_costs[mito_indexes][ii]
                
                mito_voxels = seg_mito[mito_indexes]
                min_dist_mito_type = mito_voxels[ii]
                min_dist_zyx = Mito_coordinates[ii]                                 
                
                if i*new_target_distance == np.inf:
                    errors_from_scale.append(Synapse)
                else:
                    assert i*new_target_distance != np.inf
                    #print(min_dist_mito_type)
                    dist_lst.append((i*new_target_distance, min_dist_mito_type))
    else:
        errors_from_no_mito.append(Synapse)
    
    return dist_lst, errors_from_synapse_loc, errors_from_no_mito, errors_from_scale


def Attribute_Creator(Body_ID, counts1, counts2, counts3, inf_dists):
    Neuro_Attr = counts1 + counts2 + counts3 + inf_dists[int(Body_ID)]
    return Neuro_Attr

#Main Function

def process_neuron_mito(ID):
    Mito_Dists = {}
    inf_dists = {}
#Sample_Dists = {}
    #Preprocessing before Mito_Synapse_Distance
    Body_ID = int(ID)
    synapses, num_synapses = preprocess(Body_ID)
    dists = []
    errors_syn = []
    errors_mito = []
    errors_scale = []
    bodies = []
    headers = ['x', 'y', 'z', 'dist', 'mito type']

    for j in range(len(synapses)):#[24477, 19905, 15340]]:
        Synapse = list(synapses[j])
        #print(Synapse)
        scale = 3
        i = 2**scale
        center_zyx = np.array(Synapse[::-1])
        boxsize1D = 250
        box = np.array([center_zyx//i - boxsize1D//i, center_zyx//i + boxsize1D//i])
        vol = fetch_labelmap_voxels('emdata4.int.janelia.org:8900', '5696', 'segmentation', box, scale = scale)
        if len(vol) % 2 == 0:
            center = len(vol)//2
        else:
            center = len(vol) // 2 - 1
        mito = fetch_labelmap_voxels('emdata4.int.janelia.org:8900', '5696', 'mito_20190501.24734943', box, scale = scale)    
        seg_mask = (vol == Body_ID)
        seg_mask, local_bb = boundary_box(seg_mask)
        #print(local_bb)
        ((z0, y0, x0),(z1,y1,x1)) = local_bb
        seg_vol = seg_mask[z0:z1+1, y0:y1+1, x0:x1+1]
        seg_mito = mito[z0:z1+1, y0:y1+1, x0:x1+1]
        local_syn = [center - z0, center - y0, center - x0]
        #print(is_synapse_neuron(seg_vol, [local_syn]))
        dist, error_syn, error_mito, error_scale = Mito_Synapse_Distance([local_syn], seg_vol, seg_mito, Synapse)
        #print(local_syn)
        dists.append(dist)
        if error_syn != []:
            errors_syn.append((error_syn, Body_ID))
        if error_mito != []:
            errors_mito.append((error_mito, Body_ID))
        if error_scale != []:
            errors_scale.append((error_scale, Body_ID))
        if dist != []:
            bodies.append([str(x) for x in tuple(Synapse) + dist[0]])
        else:
            bodies.append([str(x) for x in tuple(Synapse) + (0, 0)])

    with open(f'{Body_ID}_synapse_info.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(headers)
        writer.writerows(bodies)

    Mito_Dists.update({Body_ID:dists})
    inf_dists.update({Body_ID: [len(errors_mito) + len(errors_scale)]})

    Mito_segs = {}
    mito1, mito2, mito3 = Mito_segregation(Mito_Dists, ID)
    Mito_segs.update({ID:(mito1, mito2, mito3)})
    bins = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 5000]

    All_Neuro_Attr = {}

    if len(Mito_segs[ID][0]) == 0:
        counts1 = [0 for x in range(len(bins)-1)]
    else:
        a = np.hstack(Mito_segs[ID][0])
        counts1, bin_edges = np.histogram(a, bins= bins)
        counts1 = counts1.tolist()

    if len(Mito_segs[ID][1]) == 0:
        counts2 = [0 for x in range(len(bins)-1)]
    else:
        b = np.hstack(Mito_segs[ID][1])
        counts2, bin_edges = np.histogram(b, bins= bins)
        counts2 = counts2.tolist()

    if len(Mito_segs[ID][2]) == 0:
        counts3 = [0 for x in range(len(bins)-1)]
    else:
        c = np.hstack(Mito_segs[ID][2])
        counts3, bin_edges = np.histogram(c, bins= bins)
        counts3 = counts3.tolist()
        
    All_Neuro_Attr.update({Body_ID: Attribute_Creator(Body_ID, counts1, counts2, counts3, inf_dists)})
    with open(f'{Body_ID}_dimensions.json', 'w') as outfile:
        json.dump(All_Neuro_Attr, outfile)

    return True


if __name__ == "__main__":
    main()
