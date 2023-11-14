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

#Load the trajectory of the system
t = md.load('../Data/traj_copo45perc_100frames.gro')
print(t)


#Load the stereochemistry of the system
with open('./sequence_copolymer_45mer') as f1:
     lines = (line for line in f1 if not line.startswith('#'))
     chemistry = np.loadtxt(lines, delimiter=' ',skiprows=0,usecols = (0),dtype=("str"))  


#compute the vectors for each monomer type
def SLS_vectors(index,tar_list,Coord):
    b1=Coord[5+index]-Coord[index]
    b2=Coord[1+index]-Coord[index]
    b3=Coord[3+index]-Coord[1+index]
    b4=Coord[2+index]-Coord[1+index]
    b5=Coord[4+index]-Coord[2+index]
    b6=Coord[6+index]-Coord[2+index]
    b7=Coord[7+index]-Coord[6+index]
    b8=Coord[8+index]-Coord[6+index]
    b9=Coord[9+index]-Coord[6+index]
    b10=Coord[10+index]-Coord[2+index]
   
    tar_list.append([b1[0],b1[1],b1[2]])
    tar_list.append([b2[0],b2[1],b2[2]])
    tar_list.append([b3[0],b3[1],b3[2]])      
    tar_list.append([b4[0],b4[1],b4[2]])
    tar_list.append([b5[0],b5[1],b5[2]])
    tar_list.append([b6[0],b6[1],b6[2]])
    tar_list.append([b7[0],b7[1],b7[2]])      
    tar_list.append([b8[0],b8[1],b8[2]])
    tar_list.append([b9[0],b9[1],b9[2]])
    tar_list.append([b10[0],b10[1],b10[2]])
 
    return      


def SLM_vectors(index,tar_list,Coord):
    b1=Coord[1+index]-Coord[index]
    b2=Coord[3+index]-Coord[1+index]
    b3=Coord[2+index]-Coord[1+index]
    b4=Coord[4+index]-Coord[2+index]
    b5=Coord[5+index]-Coord[2+index]
    b6=Coord[6+index]-Coord[5+index]
    b7=Coord[7+index]-Coord[5+index]
    b8=Coord[8+index]-Coord[5+index]
    b9=Coord[9+index]-Coord[2+index]
   
    tar_list.append([b1[0],b1[1],b1[2]])
    tar_list.append([b2[0],b2[1],b2[2]])
    tar_list.append([b3[0],b3[1],b3[2]])      
    tar_list.append([b4[0],b4[1],b4[2]])
    tar_list.append([b5[0],b5[1],b5[2]])
    tar_list.append([b6[0],b6[1],b6[2]])
    tar_list.append([b7[0],b7[1],b7[2]])      
    tar_list.append([b8[0],b8[1],b8[2]])
    tar_list.append([b9[0],b9[1],b9[2]])
  
    return      

def SLE_vectors(index,tar_list,Coord):
    b1=Coord[1+index]-Coord[index]
    b2=Coord[3+index]-Coord[1+index]
    b3=Coord[2+index]-Coord[1+index]
    b4=Coord[4+index]-Coord[2+index]
    b5=Coord[5+index]-Coord[2+index]
    b6=Coord[6+index]-Coord[5+index]
    b7=Coord[7+index]-Coord[2+index]
    b8=Coord[8+index]-Coord[7+index]
    b9=Coord[9+index]-Coord[7+index]
    b10=Coord[10+index]-Coord[7+index]
  
    tar_list.append([b1[0],b1[1],b1[2]])
    tar_list.append([b2[0],b2[1],b2[2]])
    tar_list.append([b3[0],b3[1],b3[2]])      
    tar_list.append([b4[0],b4[1],b4[2]])
    tar_list.append([b5[0],b5[1],b5[2]])
    tar_list.append([b6[0],b6[1],b6[2]])
    tar_list.append([b7[0],b7[1],b7[2]])      
    tar_list.append([b8[0],b8[1],b8[2]])
    tar_list.append([b9[0],b9[1],b9[2]])
    tar_list.append([b10[0],b10[1],b10[2]])

    return      


#Perform the encoding to exctract the input and target of the Deep Learning model
def encoding(frames,save_path):  
 input_file=[]
 target_file=[]  
 for frame_number, frameIndx in enumerate(frames):
    LXX,LYY,LZZ=t.unitcell_lengths[frameIndx][0],t.unitcell_lengths[frameIndx][1],t.unitcell_lengths[frameIndx][2]
    hLXX,hLYY,hLZZ=LXX/2.0,LYY/2.0,LZZ/2.0
    Coord=np.zeros([npart,3],dtype=np.float32)
    frame_counter_1 = -1
    frame_counter_2 = 0
    for j in range(nchain):
        comx,comy,comz=0.0,0.0,0.0
        coordCG=np.zeros([nmer,3],dtype=np.float32)
        imageCG=np.zeros([nmer,3],dtype=np.float32)
        inp_list = []
        tar_list = []
        
        for jj in range(nmer):
            posCM=[0.,0.,0.]
            if(jj==0): masses,merlen = masses_SLS,merlen_SLS
            elif(jj==nmer-1): masses,merlen = masses_SLE,merlen_SLE
            else: masses,merlen = masses_SLM,merlen_SLM
            totmass=np.sum(masses)
            for ii in range(merlen):
                frame_counter_1 += 1
                partIndx = frame_counter_1
                Coord[partIndx]=t.xyz[frameIndx,(partIndx),:]
                if (jj!=0 or ii!=0):
                    if(Coord[partIndx][0]-Coord[partIndx-1][0]<-hLXX): Coord[partIndx][0]+=LXX
                    if(Coord[partIndx][0]-Coord[partIndx-1][0]>hLXX): Coord[partIndx][0]-=LXX
                    if(Coord[partIndx][1]-Coord[partIndx-1][1]<-hLYY): Coord[partIndx][1]+=LYY
                    if(Coord[partIndx][1]-Coord[partIndx-1][1]>hLYY): Coord[partIndx][1]-=LYY
                    if(Coord[partIndx][2]-Coord[partIndx-1][2]<-hLZZ): Coord[partIndx][2]+=LZZ
                    if(Coord[partIndx][2]-Coord[partIndx-1][2]>hLZZ): Coord[partIndx][2]-=LZZ

                posCM[0]+=Coord[partIndx][0]*masses[ii]
                posCM[1]+=Coord[partIndx][1]*masses[ii]
                posCM[2]+=Coord[partIndx][2]*masses[ii]

            posCM[0]/=totmass
            posCM[1]/=totmass
            posCM[2]/=totmass
            coordCG[jj]=posCM[:]
        
        for jj in range(nmer):
            if(jj==0):
                merlen = merlen_SLS
                SLS_vectors(frame_counter_2,tar_list,Coord)
            elif(jj==nmer-1): 
                merlen = merlen_SLE
                SLE_vectors(frame_counter_2,tar_list,Coord)
            else:
                merlen = merlen_SLM  
                SLM_vectors(frame_counter_2,tar_list,Coord)
            imageCG[jj] = [coordCG[jj][0],coordCG[jj][1],coordCG[jj][2]]
            frame_counter_2 += merlen

        tar_list = np.array(tar_list,dtype=np.float64)                
        nimages = int(np.ceil((tar_list.shape[0])/IMG_WIDTH))  
        s = int(np.ceil(nmer/nimages)) 
        meridx = [(s*(h+1)-1) for h in range(nimages)]
        meridx.pop(-1)
        meridx.append(nmer-1)
        AT = []
        CG = []
        frame_counter = -1
        arrAT = np.zeros([IMG_WIDTH,3],dtype=np.float64)
        arrCG = np.zeros([IMG_WIDTH,7],dtype=np.float64)
        c = -1
        for jj in range(nmer):
            if(jj==0): 
                dtseg = spots_SLS
                m_id = [0,0,0,1]
            elif(jj==nmer-1):
                dtseg = spots_SLE
                m_id = [1,0,0,0]
            elif chemistry[jj]=="L":
                dtseg = spots_SLM
                m_id = [0,1,0,0]
            elif chemistry[jj]=="D":
                dtseg = spots_SLM
                m_id = [0,0,1,0]
            for g in range(dtseg):
                frame_counter += 1
                AT.append([tar_list[frame_counter][0],tar_list[frame_counter][1],tar_list[frame_counter][2]])
                CG.append([imageCG[jj][0],imageCG[jj][1],imageCG[jj][2],m_id[0],m_id[1],m_id[2],m_id[3]])
 
            if jj in meridx: 
                c += 1
                AT = np.array(AT,dtype=np.float64)
                CG = np.array(CG,dtype=np.float64)
                shiftx = 60
                arrAT[shiftx:shiftx+int(AT.shape[0])] = AT[:]
                arrCG[shiftx:shiftx+int(CG.shape[0])] = CG[:]  
                input_file.append(arrCG)
                target_file.append(arrAT)                
                AT = []
                CG = []
                arrAT = np.zeros([IMG_WIDTH,3],dtype=np.float64)
                arrCG = np.zeros([IMG_WIDTH,7],dtype=np.float64)
                
 final_input=np.array(input_file,dtype=np.float64)
 final_target=np.array(target_file,dtype=np.float64) 
 print(final_input.shape, final_target.shape)      
 print(final_target.min(),final_target.max())
 with open(save_path+'_input.pkl','wb') as f:
     pickle.dump(final_input, f)

 with open(save_path+'_target.pkl','wb') as f:
     pickle.dump(final_target, f)        
 input_file = []
 target_file = []


random.seed(100)
frames = random.sample(range(100), 100)

save_path="train" 
encoding(frames[:80],save_path)

save_path="val" 
encoding(frames[80:90],save_path) 

save_path="test" 
encoding(frames[90:],save_path) 
print(frames[90:])
