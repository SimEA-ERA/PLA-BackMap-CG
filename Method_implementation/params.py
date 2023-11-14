# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:25:55 2023

@author: e.christofi
"""

# Number of monomers per chain
nmer = 100

# Number of chains per frame
nchain = 70

# Number of atoms per frame
npart = 63210

# Number of atoms per chain
chainlen = 903

# Number of atoms for each monomer type
merlen_SLS = 10
merlen_SLM = 9
merlen_SLE = 11

# Masses of each monomer type
masses_SLS = [15.9994, 12.0110, 12.0110, 15.9994, 1.0080, 1.0080, 12.0110, 1.0080, 1.0080, 1.0080]
masses_SLM = [15.9994, 12.0110, 12.0110, 15.9994, 1.0080, 12.0110, 1.0080,1.0080,1.0080 ]
masses_SLE = [15.9994, 12.0110, 12.0110, 15.9994, 1.0080, 15.9994, 1.0080, 12.0110, 1.0080, 1.0080, 1.0080]

# Spots for each monomer type
spots_SLS = 9
spots_SLM = 9
spots_SLE = 11

# List of the test set frames
com_list=[31, 19, 89, 83, 34, 17, 11, 27, 76, 57]
IMG_WIDTH = 1024
IMG_HEIGHT = 1024

shiftx = 60
EPOCHS = 1000
BATCH_SIZE=64

lbv = 1
lba = 0
lda = 0
lbl = 1
lv0 = 0
