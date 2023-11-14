# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:24:35 2023

@author: e.christofi
"""

import mdtraj as md
import numpy as np
from params import *
import random
import pickle
t = md.load('../Data/traj_copo45perc_100frames.gro')
print(t)


def encoding(frames,save_path):  
 input_file=[]
 
 for frame_number, frameIndx in enumerate(frames):
    LXX,LYY,LZZ=t.unitcell_lengths[frameIndx][0],t.unitcell_lengths[frameIndx][1],t.unitcell_lengths[frameIndx][2]
    input_file.append([LXX,LYY,LZZ])
                
 final_input=np.array(input_file,dtype=np.float64)
 print(final_input.shape)      
 
 with open(save_path+'_boxes.pkl','wb') as f:
     pickle.dump(final_input, f)



random.seed(100)
frames = random.sample(range(100), 100)

save_path="test" 
encoding(frames[90:],save_path) 
print(frames[90:])
