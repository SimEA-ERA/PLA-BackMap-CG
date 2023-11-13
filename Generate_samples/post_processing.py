# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:15:00 2023

@author: e.christofi
"""

import numpy as np
import os
import pickle
from params import *
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import time
import scipy.stats as stats
#mute the warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 20})

#Load the itp file
bonds_comp = np.loadtxt("./100LA_pairs.itp", comments=';',skiprows=910,max_rows=905,usecols=[0,1],dtype=int)
bonds_comp -= 1
angles_comp = np.loadtxt("./100LA_pairs.itp", comments=';',skiprows=1818,max_rows=1602,usecols=[0,1,2],dtype=int)
angles_comp -= 1
dihedrals_comp = np.loadtxt("./100LA_pairs.itp", comments=';',skiprows=3422,max_rows=2000,usecols=[0,1,2,3],dtype=int)
dihedrals_comp -= 1


#Calculate bond lengths
def dist(p1,p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)**0.5

#Calculate bond angles
def angle(p1,p2,p3):
    vec1=[p1[0]-p2[0],p1[1]-p2[1],p1[2]-p2[2]]
    vec2=[p3[0]-p2[0],p3[1]-p2[1],p3[2]-p2[2]]
    return np.arccos(np.dot(vec1,vec2)/np.sqrt(np.dot(vec1,vec1))/np.sqrt(np.dot(vec2,vec2)))/np.pi*180.0

#Calculate dihedral angles
def dihedral(p1,p2,p3,p4):
    b0=[p1[0]-p2[0],p1[1]-p2[1],p1[2]-p2[2]]
    b1=[p3[0]-p2[0],p3[1]-p2[1],p3[2]-p2[2]]
    b2=[p4[0]-p3[0],p4[1]-p3[1],p4[2]-p3[2]]

    b1/=np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

# Calculate the bonds, angles, and dihedrals for a given configuration
def calc_dists(idx_frame,data):
  frame = idx_frame   

  Pbond_lengths = defaultdict(list)  
  Pbond_angles = defaultdict(list)  
  Pdihedral_angles = defaultdict(list)  

  Tbond_lengths = defaultdict(list)  
  Tbond_angles = defaultdict(list)  
  Tdihedral_angles = defaultdict(list)  


#Load the Initial prediction and target configurations
  with open(data+'/PStart_'+str(frame)+'.dat') as f1:
     lines = (line for line in f1 if not line.startswith('#'))
     Pdata = np.loadtxt(lines, delimiter=' ',skiprows=2,usecols = (1,2,3),dtype=(np.float_))   

  with open(data+'/PStart_'+str(frame)+'.dat') as f1:
     lines = (line for line in f1 if not line.startswith('#'))
     atom_names = np.loadtxt(lines, delimiter=' ',skiprows=2,usecols = (0),dtype=(np.string_))   

  with open(data+'/TStart_'+str(frame)+'.dat') as f1:
     lines = (line for line in f1 if not line.startswith('#'))
     Tdata = np.loadtxt(lines, delimiter=' ',skiprows=2,usecols = (1,2,3),dtype=(np.float_))   


  
  for ichain in range(nchain):
     Pres_atoms = Pdata[ichain*chainlen:(ichain+1)*chainlen]     
     Tres_atoms = Tdata[ichain*chainlen:(ichain+1)*chainlen]     
     for ibond in range(len(bonds_comp)):
         comp = bonds_comp[ibond]
         name = atom_names[comp[0]].astype(str)+"-"+atom_names[comp[1]].astype(str)          
         Pbond_val = dist(Pres_atoms[comp[0]],Pres_atoms[comp[1]])
         Pbond_lengths[name].append(Pbond_val)  
         Tbond_val = dist(Tres_atoms[comp[0]],Tres_atoms[comp[1]])
         Tbond_lengths[name].append(Tbond_val)  

     for iangle in range(len(angles_comp)):
         comp = angles_comp[iangle]
         name = atom_names[comp[0]].astype(str)+"-"+atom_names[comp[1]].astype(str)+"-"+atom_names[comp[2]].astype(str)          
         Pangle_val = angle(Pres_atoms[comp[0]],Pres_atoms[comp[1]],Pres_atoms[comp[2]])
         Pbond_angles[name].append(Pangle_val) 
         Tangle_val = angle(Tres_atoms[comp[0]],Tres_atoms[comp[1]],Tres_atoms[comp[2]])
         Tbond_angles[name].append(Tangle_val) 

     for idihedral in range(len(dihedrals_comp)):
         comp = dihedrals_comp[idihedral]
         name = atom_names[comp[0]].astype(str)+"-"+atom_names[comp[1]].astype(str)+"-"+atom_names[comp[2]].astype(str) +"-"+atom_names[comp[3]].astype(str)           
         Pdihedral_val = dihedral(Pres_atoms[comp[0]],Pres_atoms[comp[1]],Pres_atoms[comp[2]],Pres_atoms[comp[3]])
         Pdihedral_angles[name].append(Pdihedral_val) 
         Tdihedral_val = dihedral(Tres_atoms[comp[0]],Tres_atoms[comp[1]],Tres_atoms[comp[2]],Tres_atoms[comp[3]])
         Tdihedral_angles[name].append(Tdihedral_val) 
  
  return [Pbond_lengths,Pbond_angles,Pdihedral_angles,Tbond_lengths,Tbond_angles,Tdihedral_angles]

def plot_dists(frame_idx):
  data = './Data'

  totalA_z = 0
  totalB_z = 0
  totalC_z = 0

  totalA = 0
  totalB = 0
  totalC = 0

  bl_p = defaultdict(list)
  ba_p = defaultdict(list)
  da_p = defaultdict(list)

  bl_t = defaultdict(list)
  ba_t = defaultdict(list)
  da_t = defaultdict(list)
   
  results = calc_dists(frame_idx,data)
  pool.close()
   
 

  for i_dic in list(results[0]):
              bl_p[i_dic] += results[0][i_dic]
              bl_o[i_dic] += results[3][i_dic]
  for i_dic in list(results[1]):
              ba_p[i_dic] += results[1][i_dic]
              ba_o[i_dic] += results[4][i_dic]
  for i_dic in list(results[2]):
              da_p[i_dic] += results[2][i_dic]
              da_o[i_dic] += results[5][i_dic]

  
  comp_bond_lengths = list(bl_o) 
  comp_bond_angles = list(ba_o) 
  comp_dihedral_angles = list(da_o) 

  for i in comp_bond_lengths:
   hist1, bin_edges1 = np.histogram(bl_o[i],bins=40,range=(0.05,0.25),density=True)
   hist3, bin_edges3 = np.histogram(bl_p[i],bins=40,range=(0.05,0.25),density=True)

   hist1_z = stats.zscore(hist1)
   hist3_z = stats.zscore(hist3)
   
   totalA+=(np.linalg.norm((hist1-hist3), ord=1))  
   totalA_z+=(np.linalg.norm((hist1_z-hist3_z), ord=1))
  
  totalA /= len(bl_o)      
  totalA_z /= len(bl_o)*40.    

  for i in comp_bond_angles:
   hist1, bin_edges1 = np.histogram(ba_o[i],bins=90,range=(0.0,180.),density=True)
   hist3, bin_edges3 = np.histogram(ba_p[i],bins=90,range=(0.0,180.),density=True)

   hist1_z = stats.zscore(hist1)
   hist3_z = stats.zscore(hist3)
   
   totalB+=(np.linalg.norm((hist1-hist3), ord=1))
   totalB_z+=(np.linalg.norm((hist1_z-hist3_z), ord=1))

  totalB /= len(ba_o)   
  totalB_z /= len(ba_o)*90. 

  for i in comp_dihedral_angles:
   hist1, bin_edges1 = np.histogram(da_o[i],bins=180,range=(-180.,180.),density=True)
   hist3, bin_edges3 = np.histogram(da_p[i],bins=180,range=(-180.,180.),density=True)

   hist1_z = stats.zscore(hist1)
   hist3_z = stats.zscore(hist3)

   totalC+=(np.linalg.norm((hist1-hist3), ord=1))   
   totalC_z+=(np.linalg.norm((hist1_z-hist3_z), ord=1))
  
  totalC /= len(da_o) 
  totalC_z /= len(da_o)*180. 
  
  nframes = len(frames)
  totalA_z /= nframes 
  totalB_z /= nframes
  totalC_z /= nframes
  total_psi_z = totalA_z + totalB_z + totalC_z

  totalA /= nframes 
  totalB /= nframes
  totalC /= nframes
  total_psi = totalA + totalB + totalC


  print("AT psi_z:",epoch,float(totalA_z),float(totalB_z),float(totalC_z),float(total_psi_z))
  print("AT psi:",epoch,float(totalA),float(totalB),float(totalC),float(total_psi))

#Plot the distributions 
  os.mkdir("./bond_lengths")
  os.mkdir("./bond_angles")
  os.mkdir("./dihedral_angles")
  print("Plotting the distribution plots")
  print("Plotting bond angles")
  for i in comp_bond_angles:
        fig,ax = plt.subplots()
        sns.histplot(ba_o[i],bins=90,binrange=(0.0,180.), stat="density",  ax=ax, color="red", label="Target", fill=False,element="poly")
        sns.histplot(ba_p[i],bins=90,binrange=(0.0,180.), stat="density",  ax=ax, color="blue", label="Prediction", fill=False,element="poly")        
        plt.title(label='Bond angle: '+i)
        plt.xlabel('bond angle (degrees)')
        plt.ylabel('probability density')
        plt.legend(fontsize=12) 
        plt.savefig('./bond_angles/{}.png'.format(i), bbox_inches='tight')
        plt.close()
  print("Plotting bond lengths")  
  for i in comp_bond_lengths:
       fig,ax = plt.subplots()
       sns.histplot(bl_o[i],bins=40, stat="density",binrange=(0.05,0.25),  ax=ax, color="red", label="Target", fill=False,element="poly")
       sns.histplot(bl_p[i],bins=40, stat="density",binrange=(0.05,0.25),  ax=ax, color="blue", label="Prediction", fill=False,element="poly")       
       plt.title(label='Bond length: '+i)
       plt.xlabel('bond length (nm)')
       plt.ylabel('probability density')
       plt.legend(fontsize=12) 
       plt.savefig('./bond_lengths/{}.png'.format(i), bbox_inches='tight')
       plt.close()  
  print("Plotting dihedral angles")  
  for i in comp_dihedral_angles:
       fig,ax = plt.subplots()
       sns.histplot(da_o[i],bins=180,binrange=(-180.,180.), stat="density",  ax=ax, color="red", label="Target", fill=False,element="poly")
       sns.histplot(da_p[i],bins=180,binrange=(-180.,180.), stat="density",  ax=ax, color="blue", label="Prediction", fill=False,element="poly")       
       plt.title(label='Dihedral angle: '+i)
       plt.xlabel('dihedral angle (degrees)')
       plt.ylabel('probability density')
       plt.legend(fontsize=12) 
       plt.savefig('./dihedral_angles/{}.png'.format(i), bbox_inches='tight')
       plt.close()        
        
         
   
         
   