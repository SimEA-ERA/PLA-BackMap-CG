# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:03:27 2023

@author: e.christofi
"""

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from params import *
import pickle
import time
import multiprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.experimental.set_synchronous_execution(enable=False)
    

PATH = "."

with open(PATH+'/test_target.pkl','rb') as f:
    test_target = pickle.load(f)
    print(test_target.shape)

with open(PATH+'/test_input.pkl','rb') as f:
    test_input = pickle.load(f)
    print(test_input.shape)

with open(PATH+'/test_boxes.pkl','rb') as f:
    test_boxes = pickle.load(f)
    print(test_boxes.shape)

        

test_input = test_input.reshape((int(test_input.shape[0]/nchain),nchain,test_input.shape[1],test_input.shape[2])) 
print(test_input.shape)   

test_target = test_target.reshape((int(test_target.shape[0]/nchain),nchain,test_target.shape[1],test_target.shape[2])) 
print(test_target.shape) 

    
with open('../Data_loss/chain_ba_list.pkl','rb') as f:
    chain_ba_list = pickle.load(f)
    print(len(chain_ba_list))

with open('../Data_loss/chain_da_list.pkl','rb') as f:
    chain_da_list = pickle.load(f)
    print(len(chain_da_list))

with open('../Data_loss/chain_da_sign_list.pkl','rb') as f:
    chain_da_sign_list = pickle.load(f)
    print(len(chain_da_sign_list))

    
chain_ba_array = np.array(chain_ba_list)
chain_ba_array += shiftx
chain_ba_list_1 = list(chain_ba_array[:,0])
chain_ba_list_2 = list(chain_ba_array[:,1])


chain_da_array = np.array(chain_da_list)
chain_da_array += shiftx
chain_da_list_1 = list(chain_da_array[:,0])
chain_da_list_2 = list(chain_da_array[:,1])
chain_da_list_3 = list(chain_da_array[:,2])

chain_da_sign_array = np.array(chain_da_sign_list)
chain_da_sign_list_1 = chain_da_sign_array[:,0]
chain_da_sign_list_1=tf.cast(chain_da_sign_list_1,dtype="double")
chain_da_sign_list_2 = chain_da_sign_array[:,1]
chain_da_sign_list_2=tf.cast(chain_da_sign_list_2,dtype="double")
chain_da_sign_list_3 = chain_da_sign_array[:,2]
chain_da_sign_list_3=tf.cast(chain_da_sign_list_3,dtype="double")


def downsample(entered_input,filters, size, apply_batchnorm=True,strides=2):
  conv1 = tf.keras.layers.Conv1D(filters, size, strides=strides, padding='same',use_bias=False)(entered_input)
  conv1 = tf.keras.layers.LeakyReLU()(conv1)
  
  if apply_batchnorm:
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
  
  

  return conv1


def upsample(entered_input,filters, size, skip_layer, apply_dropout=False, strides=2, apply_skip=True):
  tran1 = tf.keras.layers.Conv1DTranspose(filters, size, strides=strides,
                                    padding='same',
                                    use_bias=True)(entered_input)
  tran1 = tf.keras.layers.ReLU()(tran1) 
  if apply_dropout:
      tran1 = tf.keras.layers.Dropout(0.5)(tran1)
  
  if apply_skip:
      tran1 = tf.keras.layers.Concatenate()([tran1,skip_layer])
  return tran1


def Generator(): 
  input1 = tf.keras.layers.Input([1024,7])  
  output1 = downsample(input1, 64, 3)
  output2 = downsample(output1, 128, 3)
  output3 = downsample(output2, 256, 3)  
  output4 = downsample(output3, 512, 3) 
  output5 = downsample(output4, 512, 3) 

  output = upsample(output5, 512, 3, output4, apply_dropout=True)
  output = upsample(output, 256, 3, output3, apply_dropout=False)
  output = upsample(output, 128, 3, output2, apply_dropout=False)
  output = upsample(output, 64, 3, output1, apply_dropout=False)
  
  output = tf.keras.layers.Conv1DTranspose(64, 3, strides=2, padding="same",  activation="relu")(output)
  out = tf.keras.layers.Conv1DTranspose(3, 3, strides=1, padding="same",  activation="tanh")(output)

  model = tf.keras.models.Model(input1,out)
  return model


model_gen = Generator()
model_gen.summary()

MAE = tf.keras.losses.MeanAbsoluteError()

def bond_lengths_loss(y_p,y_t):
    p_length = tf.sqrt(tf.reduce_sum(tf.square(y_p),axis=-1))
    t_length = tf.sqrt(tf.reduce_sum(tf.square(y_t),axis=-1))  
    return MAE(t_length,p_length)   

def calc_ba(bond_1,bond_2):
    bond_lengths_1 = tf.sqrt(tf.reduce_sum(tf.square(bond_1),axis=-1))
    bond_lengths_2 = tf.sqrt(tf.reduce_sum(tf.square(bond_2),axis=-1))
    bond_lengths_prod = tf.math.multiply(bond_lengths_1, bond_lengths_2)
    
    bond_inner = -tf.reduce_sum(tf.math.multiply(bond_1,bond_2),axis=-1)
    ba = tf.math.divide_no_nan(bond_inner,bond_lengths_prod) 

    return ba


def calc_da(bond_1,bond_2,bond_3):    
    bond_cross_1 = tf.linalg.cross(bond_1,bond_2)    
    bond_cross_2 = tf.linalg.cross(bond_2,bond_3)
    cross_inner = tf.reduce_sum(tf.math.multiply(bond_cross_1, bond_cross_2),axis=-1)
  
    bond_lengths_1 = tf.sqrt(tf.reduce_sum(tf.square(bond_cross_1),axis=-1))
    bond_lengths_2 = tf.sqrt(tf.reduce_sum(tf.square(bond_cross_2),axis=-1))
    
    cross_lengths_prod = tf.math.multiply(bond_lengths_1, bond_lengths_2)
    
    da = tf.math.divide_no_nan(cross_inner,cross_lengths_prod) 

    return da


def bond_angles_loss(y_p,y_t,chain_ba_list_1,chain_ba_list_2):
    p_chain_ba_list_1 = tf.gather(y_p,chain_ba_list_1,axis=1)
    p_chain_ba_list_2 = tf.gather(y_p,chain_ba_list_2,axis=1)
    
    t_chain_ba_list_1 = tf.gather(y_t,chain_ba_list_1,axis=1)
    t_chain_ba_list_2 = tf.gather(y_t,chain_ba_list_2,axis=1)
      
    p_ba = calc_ba(p_chain_ba_list_1,p_chain_ba_list_2)
    t_ba = calc_ba(t_chain_ba_list_1,t_chain_ba_list_2)
    return MAE(t_ba,p_ba)   


def dihedral_angles_loss(y_p,y_t,chain_da_list_1,chain_da_list_2,chain_da_list_3,chain_da_sign_list_1,chain_da_sign_list_2,chain_da_sign_list_3):
    p_chain_da_list_1 = tf.gather(y_p,chain_da_list_1,axis=1)
    p_chain_da_list_2 = tf.gather(y_p,chain_da_list_2,axis=1)
    p_chain_da_list_3 = tf.gather(y_p,chain_da_list_3,axis=1)
    
    p_chain_da_list_1 = tf.math.multiply(p_chain_da_list_1 ,tf.cast(chain_da_sign_list_1[tf.newaxis,:,tf.newaxis],dtype=tf.float32))
    p_chain_da_list_2 = tf.math.multiply(p_chain_da_list_2 ,tf.cast(chain_da_sign_list_2[tf.newaxis,:,tf.newaxis],dtype=tf.float32))
    p_chain_da_list_3 = tf.math.multiply(p_chain_da_list_3 ,tf.cast(chain_da_sign_list_3[tf.newaxis,:,tf.newaxis],dtype=tf.float32))
        
    t_chain_da_list_1 = tf.gather(y_t,chain_da_list_1,axis=1)
    t_chain_da_list_2 = tf.gather(y_t,chain_da_list_2,axis=1)
    t_chain_da_list_3 = tf.gather(y_t,chain_da_list_3,axis=1)
    
    t_chain_da_list_1 = tf.math.multiply(t_chain_da_list_1 ,tf.cast(chain_da_sign_list_1[tf.newaxis,:,tf.newaxis],dtype=tf.float32))
    t_chain_da_list_2 = tf.math.multiply(t_chain_da_list_2 ,tf.cast(chain_da_sign_list_2[tf.newaxis,:,tf.newaxis],dtype=tf.float32))
    t_chain_da_list_3 = tf.math.multiply(t_chain_da_list_3 ,tf.cast(chain_da_sign_list_3[tf.newaxis,:,tf.newaxis],dtype=tf.float32))
    
    p_da = calc_da(p_chain_da_list_1,p_chain_da_list_2,p_chain_da_list_3)
    t_da = calc_da(t_chain_da_list_1,t_chain_da_list_2,t_chain_da_list_3)
    return MAE(t_da,p_da)   

def calc_v0_SLS(output_tensor,input_tensor):
  indx_b1 = tf.where(tf.equal(input_tensor[:,:,-1],1))[0::9]
  indx_b2 = tf.where(tf.equal(input_tensor[:,:,-1],1))[1::9]
  indx_b3 = tf.where(tf.equal(input_tensor[:,:,-1],1))[2::9]
  indx_b4 = tf.where(tf.equal(input_tensor[:,:,-1],1))[3::9]
  indx_b5 = tf.where(tf.equal(input_tensor[:,:,-1],1))[4::9]
  indx_b6 = tf.where(tf.equal(input_tensor[:,:,-1],1))[5::9]
  indx_b7 = tf.where(tf.equal(input_tensor[:,:,-1],1))[6::9]
  indx_b8 = tf.where(tf.equal(input_tensor[:,:,-1],1))[7::9]
  indx_b9 = tf.where(tf.equal(input_tensor[:,:,-1],1))[8::9]

  b1_tensor = tf.gather_nd(output_tensor,indx_b1) 
  b2_tensor = tf.gather_nd(output_tensor,indx_b2) 
  b3_tensor = tf.gather_nd(output_tensor,indx_b3) 
  b4_tensor = tf.gather_nd(output_tensor,indx_b4) 
  b5_tensor = tf.gather_nd(output_tensor,indx_b5) 
  b6_tensor = tf.gather_nd(output_tensor,indx_b6) 
  b7_tensor = tf.gather_nd(output_tensor,indx_b7) 
  b8_tensor = tf.gather_nd(output_tensor,indx_b8) 
  b9_tensor = tf.gather_nd(output_tensor,indx_b9) 
  masses = masses_SLS
  totmass = np.sum(masses)

  v0 = -((b1_tensor)*masses[5]+(b2_tensor)*masses[1]+(b2_tensor+b3_tensor)*masses[3]+(b2_tensor+b4_tensor)*masses[2]+(b2_tensor+b4_tensor+b5_tensor)*masses[4]+(b2_tensor+b4_tensor+b6_tensor)*masses[6]+(b2_tensor+b4_tensor+b6_tensor+b7_tensor)*masses[7]+(b2_tensor+b4_tensor+b6_tensor+b8_tensor)*masses[8]+(b2_tensor+b4_tensor+b6_tensor+b9_tensor)*masses[9])/totmass  
  return v0


def calc_v0_SLM(output_tensor,input_tensor):
  indx_b0 = tf.where(tf.equal(input_tensor[:,:,-2],1))[0::9]
  indx_b1 = tf.where(tf.equal(input_tensor[:,:,-2],1))[1::9]
  indx_b2 = tf.where(tf.equal(input_tensor[:,:,-2],1))[2::9]
  indx_b3 = tf.where(tf.equal(input_tensor[:,:,-2],1))[3::9]
  indx_b4 = tf.where(tf.equal(input_tensor[:,:,-2],1))[4::9]
  indx_b5 = tf.where(tf.equal(input_tensor[:,:,-2],1))[5::9]
  indx_b6 = tf.where(tf.equal(input_tensor[:,:,-2],1))[6::9]
  indx_b7 = tf.where(tf.equal(input_tensor[:,:,-2],1))[7::9]
  indx_b8 = tf.where(tf.equal(input_tensor[:,:,-2],1))[8::9]

  b0_tensor = tf.gather_nd(output_tensor,indx_b0) 
  b1_tensor = tf.gather_nd(output_tensor,indx_b1) 
  b2_tensor = tf.gather_nd(output_tensor,indx_b2) 
  b3_tensor = tf.gather_nd(output_tensor,indx_b3) 
  b4_tensor = tf.gather_nd(output_tensor,indx_b4) 
  b5_tensor = tf.gather_nd(output_tensor,indx_b5) 
  b6_tensor = tf.gather_nd(output_tensor,indx_b6) 
  b7_tensor = tf.gather_nd(output_tensor,indx_b7) 
  b8_tensor = tf.gather_nd(output_tensor,indx_b8) 
 
  masses = masses_SLM
  totmass = np.sum(masses)

  v0 = -((b0_tensor)*masses[0]+(b0_tensor+b1_tensor)*masses[1]+(b0_tensor+b1_tensor+b2_tensor)*masses[3]+(b0_tensor+b1_tensor+b3_tensor)*masses[2]+(b0_tensor+b1_tensor+b3_tensor+b4_tensor)*masses[4]+(b0_tensor+b1_tensor+b3_tensor+b5_tensor)*masses[5]+(b0_tensor+b1_tensor+b3_tensor+b5_tensor+b6_tensor)*masses[6]+(b0_tensor+b1_tensor+b3_tensor+b5_tensor+b7_tensor)*masses[7]+(b0_tensor+b1_tensor+b3_tensor+b5_tensor+b8_tensor)*masses[8])/totmass  
  return v0

def calc_v0_SLE(output_tensor,input_tensor):
  indx_b0 = tf.where(tf.equal(input_tensor[:,:,-3],1))[0::11]
  indx_b1 = tf.where(tf.equal(input_tensor[:,:,-3],1))[1::11]
  indx_b2 = tf.where(tf.equal(input_tensor[:,:,-3],1))[2::11]
  indx_b3 = tf.where(tf.equal(input_tensor[:,:,-3],1))[3::11]
  indx_b4 = tf.where(tf.equal(input_tensor[:,:,-3],1))[4::11]
  indx_b5 = tf.where(tf.equal(input_tensor[:,:,-3],1))[5::11]
  indx_b6 = tf.where(tf.equal(input_tensor[:,:,-3],1))[6::11]
  indx_b7 = tf.where(tf.equal(input_tensor[:,:,-3],1))[7::11]
  indx_b8 = tf.where(tf.equal(input_tensor[:,:,-3],1))[8::11]
  indx_b9 = tf.where(tf.equal(input_tensor[:,:,-3],1))[9::11]
  indx_b10 = tf.where(tf.equal(input_tensor[:,:,-3],1))[10::11]
  
  b0_tensor = tf.gather_nd(output_tensor,indx_b0) 
  b1_tensor = tf.gather_nd(output_tensor,indx_b1) 
  b2_tensor = tf.gather_nd(output_tensor,indx_b2) 
  b3_tensor = tf.gather_nd(output_tensor,indx_b3) 
  b4_tensor = tf.gather_nd(output_tensor,indx_b4) 
  b5_tensor = tf.gather_nd(output_tensor,indx_b5) 
  b6_tensor = tf.gather_nd(output_tensor,indx_b6) 
  b7_tensor = tf.gather_nd(output_tensor,indx_b7) 
  b8_tensor = tf.gather_nd(output_tensor,indx_b8) 
  b9_tensor = tf.gather_nd(output_tensor,indx_b9) 
  b10_tensor = tf.gather_nd(output_tensor,indx_b10) 
  
  masses = masses_SLE
  totmass = np.sum(masses)

  v0 = -((b0_tensor)*masses[0]+(b0_tensor+b1_tensor)*masses[1]+(b0_tensor+b2_tensor+b1_tensor)*masses[3]+(b0_tensor+b1_tensor+b3_tensor)*masses[2]+(b0_tensor+b1_tensor+b3_tensor+b4_tensor)*masses[4]+(b0_tensor+b1_tensor+b3_tensor+b5_tensor)*masses[5]+(b0_tensor+b1_tensor+b3_tensor+b7_tensor)*masses[7]+(b0_tensor+b1_tensor+b3_tensor+b5_tensor+b6_tensor)*masses[6]+(b0_tensor+b1_tensor+b3_tensor+b7_tensor+b8_tensor)*masses[8]+(b0_tensor+b1_tensor+b3_tensor+b7_tensor+b9_tensor)*masses[9]+(b0_tensor+b1_tensor+b3_tensor+b7_tensor+b10_tensor)*masses[10])/totmass   
  return v0

def v0_loss(y_pred,y_true,x):
    t_sls = calc_v0_SLS(y_true, x)
    t_slm = calc_v0_SLM(y_true, x)
    t_sle = calc_v0_SLE(y_true, x)
   
    p_sls = calc_v0_SLS(y_pred, x)
    p_slm = calc_v0_SLM(y_pred, x)
    p_sle = calc_v0_SLE(y_pred, x)
    
    return MAE(t_sls,p_sls) + MAE(t_slm,p_slm) + MAE(t_sle,p_sle)

class u_net(tf.keras.Model):
    def __init__(self, generator ,chain_ba_list_1=chain_ba_list_1 ,chain_ba_list_2=chain_ba_list_2,chain_da_list_1=chain_da_list_1 ,chain_da_list_2=chain_da_list_2,chain_da_list_3=chain_da_list_3,chain_da_sign_list_1=chain_da_sign_list_1 ,chain_da_sign_list_2=chain_da_sign_list_2,chain_da_sign_list_3=chain_da_sign_list_3,**kwargs):
        super(u_net, self).__init__(**kwargs)
        self.generator = generator()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.bl_loss_tracker = tf.keras.metrics.Mean(name="bl_loss")
        self.ba_loss_tracker = tf.keras.metrics.Mean(name="ba_loss")
        self.da_loss_tracker = tf.keras.metrics.Mean(name="da_loss")
        self.v_loss_tracker = tf.keras.metrics.Mean(name="v_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )

        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.bl_loss_tracker,
            self.da_loss_tracker,
            self.ba_loss_tracker,
            self.v_loss_tracker,
        
        ]

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred  = self.generator(x)
            reconstruction_loss = tf.keras.losses.MeanAbsoluteError()(y_pred, y_true)
            bl_loss = bond_lengths_loss(y_pred, y_true)
            ba_loss = bond_angles_loss(y_pred, y_true,chain_ba_list_1,chain_ba_list_2)
            da_loss = dihedral_angles_loss(y_pred, y_true,chain_da_list_1,chain_da_list_2,chain_da_list_3,chain_da_sign_list_1,chain_da_sign_list_2,chain_da_sign_list_3)
            v_loss = v0_loss(y_pred, y_true, x)          
            total_loss = lbv*reconstruction_loss + lbl*bl_loss + lba*ba_loss + lda*da_loss + lv0*v_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.bl_loss_tracker.update_state(bl_loss)
        self.ba_loss_tracker.update_state(ba_loss)
        self.da_loss_tracker.update_state(da_loss)
        self.v_loss_tracker.update_state(v_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.reconstruction_loss_tracker.result(),
            "bl_loss": self.bl_loss_tracker.result(),
            "ba_loss": self.ba_loss_tracker.result(),
            "da_loss": self.da_loss_tracker.result(),
            "v_loss": self.v_loss_tracker.result(),
        
        }

    def test_step(self, input_data):
      val_x, val_true = input_data # <-- Seperate X and y
      val_pred = self.generator(val_x)
      val_reconstruction_loss = tf.keras.losses.MeanAbsoluteError()(val_pred, val_true)
      val_bl_loss = bond_lengths_loss(val_pred, val_true)
      val_ba_loss = bond_angles_loss(val_pred, val_true,chain_ba_list_1,chain_ba_list_2)
      val_da_loss = dihedral_angles_loss(val_pred, val_true,chain_da_list_1,chain_da_list_2,chain_da_list_3,chain_da_sign_list_1,chain_da_sign_list_2,chain_da_sign_list_3)
      val_v_loss = v0_loss(val_pred, val_true, val_x)
      val_total_loss = lbv*val_reconstruction_loss + lbl*val_bl_loss + lba*val_ba_loss + lda*val_da_loss + lv0*val_v_loss
      return {"loss": val_total_loss,
               "rec_loss":val_reconstruction_loss,
              "bl_loss": val_bl_loss,
              "ba_loss": val_ba_loss,
              "da_loss": val_da_loss,
              "v_loss": val_v_loss,
              }

model = u_net(Generator)
model.compile(optimizer=tf.keras.optimizers.Adam())

      
      
      

def SLS_d(pvecs,iosx,iosy,iosz,Coords,global_count):
     b_vec = np.zeros([9,3])          
     b_vec[:] = pvecs
     v_vec = np.zeros([10,3])
     
     masses = masses_SLS
     totmass = sum(masses)
        
     v_vec[0,0] = -((b_vec[0,0])*masses[5]+(b_vec[1,0])*masses[1]+(b_vec[1,0]+b_vec[2,0])*masses[3]+(b_vec[1,0]+b_vec[3,0])*masses[2]+(b_vec[1,0]+b_vec[3,0]+b_vec[4,0])*masses[4]+(b_vec[1,0]+b_vec[3,0]+b_vec[5,0])*masses[6]+(b_vec[1,0]+b_vec[3,0]+b_vec[5,0]+b_vec[6,0])*masses[7]+(b_vec[1,0]+b_vec[3,0]+b_vec[5,0]+b_vec[7,0])*masses[8]+(b_vec[1,0]+b_vec[3,0]+b_vec[5,0]+b_vec[8,0])*masses[9])/totmass  
     v_vec[0,1] = -((b_vec[0,1])*masses[5]+(b_vec[1,1])*masses[1]+(b_vec[1,1]+b_vec[2,1])*masses[3]+(b_vec[1,1]+b_vec[3,1])*masses[2]+(b_vec[1,1]+b_vec[3,1]+b_vec[4,1])*masses[4]+(b_vec[1,1]+b_vec[3,1]+b_vec[5,1])*masses[6]+(b_vec[1,1]+b_vec[3,1]+b_vec[5,1]+b_vec[6,1])*masses[7]+(b_vec[1,1]+b_vec[3,1]+b_vec[5,1]+b_vec[7,1])*masses[8]+(b_vec[1,1]+b_vec[3,1]+b_vec[5,1]+b_vec[8,1])*masses[9])/totmass   
     v_vec[0,2] = -((b_vec[0,2])*masses[5]+(b_vec[1,2])*masses[1]+(b_vec[1,2]+b_vec[2,2])*masses[3]+(b_vec[1,2]+b_vec[3,2])*masses[2]+(b_vec[1,2]+b_vec[3,2]+b_vec[4,2])*masses[4]+(b_vec[1,2]+b_vec[3,2]+b_vec[5,2])*masses[6]+(b_vec[1,2]+b_vec[3,2]+b_vec[5,2]+b_vec[6,2])*masses[7]+(b_vec[1,2]+b_vec[3,2]+b_vec[5,2]+b_vec[7,2])*masses[8]+(b_vec[1,2]+b_vec[3,2]+b_vec[5,2]+b_vec[8,2])*masses[9])/totmass  
  
     v_vec[5,0] = v_vec[0,0] + b_vec[0,0]
     v_vec[5,1] = v_vec[0,1] + b_vec[0,1]
     v_vec[5,2] = v_vec[0,2] + b_vec[0,2]
     
     v_vec[1,0] = v_vec[0,0] + b_vec[1,0]
     v_vec[1,1] = v_vec[0,1] + b_vec[1,1]
     v_vec[1,2] = v_vec[0,2] + b_vec[1,2]
     
     v_vec[3,0] = v_vec[1,0] + b_vec[2,0]
     v_vec[3,1] = v_vec[1,1] + b_vec[2,1]
     v_vec[3,2] = v_vec[1,2] + b_vec[2,2]
     
     v_vec[2,0] = v_vec[1,0] + b_vec[3,0]
     v_vec[2,1] = v_vec[1,1] + b_vec[3,1]
     v_vec[2,2] = v_vec[1,2] + b_vec[3,2]
     
     v_vec[4,0] = v_vec[2,0] + b_vec[4,0]
     v_vec[4,1] = v_vec[2,1] + b_vec[4,1]
     v_vec[4,2] = v_vec[2,2] + b_vec[4,2]
          
     v_vec[6,0] = v_vec[2,0] + b_vec[5,0]
     v_vec[6,1] = v_vec[2,1] + b_vec[5,1]
     v_vec[6,2] = v_vec[2,2] + b_vec[5,2]
        
     v_vec[7,0] = v_vec[6,0] + b_vec[6,0]
     v_vec[7,1] = v_vec[6,1] + b_vec[6,1]
     v_vec[7,2] = v_vec[6,2] + b_vec[6,2]

     v_vec[8,0] = v_vec[6,0] + b_vec[7,0]
     v_vec[8,1] = v_vec[6,1] + b_vec[7,1]
     v_vec[8,2] = v_vec[6,2] + b_vec[7,2]

     v_vec[9,0] = v_vec[6,0] + b_vec[8,0]
     v_vec[9,1] = v_vec[6,1] + b_vec[8,1]
     v_vec[9,2] = v_vec[6,2] + b_vec[8,2]


     Coords[global_count][:] = [iosx+v_vec[0,0],iosy+v_vec[0,1],iosz+v_vec[0,2]]
     Coords[global_count+1][:] = [iosx+v_vec[1,0],iosy+v_vec[1,1],iosz+v_vec[1,2]]
     Coords[global_count+2][:] = [iosx+v_vec[2,0],iosy+v_vec[2,1],iosz+v_vec[2,2]]
     Coords[global_count+3][:] = [iosx+v_vec[3,0],iosy+v_vec[3,1],iosz+v_vec[3,2]]
     Coords[global_count+4][:] = [iosx+v_vec[4,0],iosy+v_vec[4,1],iosz+v_vec[4,2]]
     Coords[global_count+5][:] = [iosx+v_vec[5,0],iosy+v_vec[5,1],iosz+v_vec[5,2]]
     Coords[global_count+6][:] = [iosx+v_vec[6,0],iosy+v_vec[6,1],iosz+v_vec[6,2]]
     Coords[global_count+7][:] = [iosx+v_vec[7,0],iosy+v_vec[7,1],iosz+v_vec[7,2]]
     Coords[global_count+8][:] = [iosx+v_vec[8,0],iosy+v_vec[8,1],iosz+v_vec[8,2]]
     Coords[global_count+9][:] = [iosx+v_vec[9,0],iosy+v_vec[9,1],iosz+v_vec[9,2]]
         
     return 
      

def SLM_d(pvecs,iosx,iosy,iosz,Coords,global_count):
     b_vec = np.zeros([9,3])          
     b_vec[:] = pvecs
     
     v_vec = np.zeros([9,3])
     
     masses = masses_SLM
     totmass = sum(masses)
        
     v_vec[0,0] = -((b_vec[0,0])*masses[0]+(b_vec[0,0]+b_vec[1,0])*masses[1]+(b_vec[0,0]+b_vec[2,0]+b_vec[1,0])*masses[3]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0])*masses[2]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0]+b_vec[4,0])*masses[4]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0]+b_vec[5,0])*masses[5]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0]+b_vec[5,0]+b_vec[6,0])*masses[6]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0]+b_vec[5,0]+b_vec[7,0])*masses[7]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0]+b_vec[5,0]+b_vec[8,0])*masses[8])/totmass  + b_vec[0,0]
     v_vec[0,1] = -((b_vec[0,1])*masses[0]+(b_vec[0,1]+b_vec[1,1])*masses[1]+(b_vec[0,1]+b_vec[2,1]+b_vec[1,1])*masses[3]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1])*masses[2]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1]+b_vec[4,1])*masses[4]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1]+b_vec[5,1])*masses[5]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1]+b_vec[5,1]+b_vec[6,1])*masses[6]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1]+b_vec[5,1]+b_vec[7,1])*masses[7]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1]+b_vec[5,1]+b_vec[8,1])*masses[8])/totmass  + b_vec[0,1]
     v_vec[0,2] = -((b_vec[0,2])*masses[0]+(b_vec[0,2]+b_vec[1,2])*masses[1]+(b_vec[0,2]+b_vec[2,2]+b_vec[1,2])*masses[3]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2])*masses[2]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2]+b_vec[4,2])*masses[4]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2]+b_vec[5,2])*masses[5]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2]+b_vec[5,2]+b_vec[6,2])*masses[6]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2]+b_vec[5,2]+b_vec[7,2])*masses[7]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2]+b_vec[5,2]+b_vec[8,2])*masses[8])/totmass  + b_vec[0,2]
    
     v_vec[1,0] = v_vec[0,0] + b_vec[1,0]
     v_vec[1,1] = v_vec[0,1] + b_vec[1,1]
     v_vec[1,2] = v_vec[0,2] + b_vec[1,2]
     
     v_vec[3,0] = v_vec[1,0] + b_vec[2,0]
     v_vec[3,1] = v_vec[1,1] + b_vec[2,1]
     v_vec[3,2] = v_vec[1,2] + b_vec[2,2]
     
     v_vec[2,0] = v_vec[1,0] + b_vec[3,0]
     v_vec[2,1] = v_vec[1,1] + b_vec[3,1]
     v_vec[2,2] = v_vec[1,2] + b_vec[3,2]
     
     v_vec[4,0] = v_vec[2,0] + b_vec[4,0] 
     v_vec[4,1] = v_vec[2,1] + b_vec[4,1] 
     v_vec[4,2] = v_vec[2,2] + b_vec[4,2] 
          
     v_vec[5,0] = v_vec[2,0] + b_vec[5,0] 
     v_vec[5,1] = v_vec[2,1] + b_vec[5,1] 
     v_vec[5,2] = v_vec[2,2] + b_vec[5,2] 
        
     v_vec[6,0] = v_vec[5,0] + b_vec[6,0] 
     v_vec[6,1] = v_vec[5,1] + b_vec[6,1] 
     v_vec[6,2] = v_vec[5,2] + b_vec[6,2] 
 
     v_vec[7,0] = v_vec[5,0] + b_vec[7,0] 
     v_vec[7,1] = v_vec[5,1] + b_vec[7,1] 
     v_vec[7,2] = v_vec[5,2] + b_vec[7,2] 

     v_vec[8,0] = v_vec[5,0] + b_vec[8,0] 
     v_vec[8,1] = v_vec[5,1] + b_vec[8,1] 
     v_vec[8,2] = v_vec[5,2] + b_vec[8,2] 
     

     Coords[global_count][:] = [iosx+v_vec[0,0],iosy+v_vec[0,1],iosz+v_vec[0,2]]
     Coords[global_count+1][:] = [iosx+v_vec[1,0],iosy+v_vec[1,1],iosz+v_vec[1,2]]
     Coords[global_count+2][:] = [iosx+v_vec[2,0],iosy+v_vec[2,1],iosz+v_vec[2,2]]
     Coords[global_count+3][:] = [iosx+v_vec[3,0],iosy+v_vec[3,1],iosz+v_vec[3,2]]
     Coords[global_count+4][:] = [iosx+v_vec[4,0],iosy+v_vec[4,1],iosz+v_vec[4,2]]
     Coords[global_count+5][:] = [iosx+v_vec[5,0],iosy+v_vec[5,1],iosz+v_vec[5,2]]
     Coords[global_count+6][:] = [iosx+v_vec[6,0],iosy+v_vec[6,1],iosz+v_vec[6,2]]
     Coords[global_count+7][:] = [iosx+v_vec[7,0],iosy+v_vec[7,1],iosz+v_vec[7,2]]
     Coords[global_count+8][:] = [iosx+v_vec[8,0],iosy+v_vec[8,1],iosz+v_vec[8,2]]
          
     return 
      

def SLE_d(pvecs,iosx,iosy,iosz,Coords,global_count):
     b_vec = np.zeros([11,3])          
     b_vec[:] = pvecs
     
     v_vec = np.zeros([11,3])
     
     masses = masses_SLE
     totmass = sum(masses)
        
     v_vec[0,0] = -((b_vec[0,0])*masses[0]+(b_vec[0,0]+b_vec[1,0])*masses[1]+(b_vec[0,0]+b_vec[2,0]+b_vec[1,0])*masses[3]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0])*masses[2]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0]+b_vec[4,0])*masses[4]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0]+b_vec[5,0])*masses[5]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0]+b_vec[7,0])*masses[7]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0]+b_vec[5,0]+b_vec[6,0])*masses[6]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0]+b_vec[7,0]+b_vec[8,0])*masses[8]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0]+b_vec[7,0]+b_vec[9,0])*masses[9]+(b_vec[0,0]+b_vec[1,0]+b_vec[3,0]+b_vec[7,0]+b_vec[10,0])*masses[10])/totmass + b_vec[0,0]  
     v_vec[0,1] = -((b_vec[0,1])*masses[0]+(b_vec[0,1]+b_vec[1,1])*masses[1]+(b_vec[0,1]+b_vec[2,1]+b_vec[1,1])*masses[3]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1])*masses[2]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1]+b_vec[4,1])*masses[4]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1]+b_vec[5,1])*masses[5]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1]+b_vec[7,1])*masses[7]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1]+b_vec[5,1]+b_vec[6,1])*masses[6]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1]+b_vec[7,1]+b_vec[8,1])*masses[8]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1]+b_vec[7,1]+b_vec[9,1])*masses[9]+(b_vec[0,1]+b_vec[1,1]+b_vec[3,1]+b_vec[7,1]+b_vec[10,1])*masses[10])/totmass + b_vec[0,1]
     v_vec[0,2] = -((b_vec[0,2])*masses[0]+(b_vec[0,2]+b_vec[1,2])*masses[1]+(b_vec[0,2]+b_vec[2,2]+b_vec[1,2])*masses[3]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2])*masses[2]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2]+b_vec[4,2])*masses[4]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2]+b_vec[5,2])*masses[5]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2]+b_vec[7,2])*masses[7]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2]+b_vec[5,2]+b_vec[6,2])*masses[6]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2]+b_vec[7,2]+b_vec[8,2])*masses[8]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2]+b_vec[7,2]+b_vec[9,2])*masses[9]+(b_vec[0,2]+b_vec[1,2]+b_vec[3,2]+b_vec[7,2]+b_vec[10,2])*masses[10])/totmass + b_vec[0,2]
     
     v_vec[1,0] = v_vec[0,0] + b_vec[1,0]
     v_vec[1,1] = v_vec[0,1] + b_vec[1,1]
     v_vec[1,2] = v_vec[0,2] + b_vec[1,2]
     
     v_vec[3,0] = v_vec[1,0] + b_vec[2,0]
     v_vec[3,1] = v_vec[1,1] + b_vec[2,1]
     v_vec[3,2] = v_vec[1,2] + b_vec[2,2]
     
     v_vec[2,0] = v_vec[1,0] + b_vec[3,0]
     v_vec[2,1] = v_vec[1,1] + b_vec[3,1]
     v_vec[2,2] = v_vec[1,2] + b_vec[3,2]
     
     v_vec[4,0] = v_vec[2,0] + b_vec[4,0] 
     v_vec[4,1] = v_vec[2,1] + b_vec[4,1] 
     v_vec[4,2] = v_vec[2,2] + b_vec[4,2] 
          
     v_vec[5,0] = v_vec[2,0] + b_vec[5,0] 
     v_vec[5,1] = v_vec[2,1] + b_vec[5,1] 
     v_vec[5,2] = v_vec[2,2] + b_vec[5,2] 
        
     v_vec[6,0] = v_vec[5,0] + b_vec[6,0] 
     v_vec[6,1] = v_vec[5,1] + b_vec[6,1] 
     v_vec[6,2] = v_vec[5,2] + b_vec[6,2] 
 
     v_vec[7,0] = v_vec[2,0] + b_vec[7,0] 
     v_vec[7,1] = v_vec[2,1] + b_vec[7,1] 
     v_vec[7,2] = v_vec[2,2] + b_vec[7,2] 

     v_vec[8,0] = v_vec[7,0] + b_vec[8,0] 
     v_vec[8,1] = v_vec[7,1] + b_vec[8,1] 
     v_vec[8,2] = v_vec[7,2] + b_vec[8,2] 

     v_vec[9,0] = v_vec[7,0] + b_vec[9,0] 
     v_vec[9,1] = v_vec[7,1] + b_vec[9,1] 
     v_vec[9,2] = v_vec[7,2] + b_vec[9,2] 

     v_vec[10,0] = v_vec[7,0] + b_vec[10,0] 
     v_vec[10,1] = v_vec[7,1] + b_vec[10,1] 
     v_vec[10,2] = v_vec[7,2] + b_vec[10,2] 
     
     Coords[global_count][:] = [iosx+v_vec[0,0],iosy+v_vec[0,1],iosz+v_vec[0,2]]
     Coords[global_count+1][:] = [iosx+v_vec[1,0],iosy+v_vec[1,1],iosz+v_vec[1,2]]
     Coords[global_count+2][:] = [iosx+v_vec[2,0],iosy+v_vec[2,1],iosz+v_vec[2,2]]
     Coords[global_count+3][:] = [iosx+v_vec[3,0],iosy+v_vec[3,1],iosz+v_vec[3,2]]
     Coords[global_count+4][:] = [iosx+v_vec[4,0],iosy+v_vec[4,1],iosz+v_vec[4,2]]
     Coords[global_count+5][:] = [iosx+v_vec[5,0],iosy+v_vec[5,1],iosz+v_vec[5,2]]
     Coords[global_count+6][:] = [iosx+v_vec[6,0],iosy+v_vec[6,1],iosz+v_vec[6,2]]
     Coords[global_count+7][:] = [iosx+v_vec[7,0],iosy+v_vec[7,1],iosz+v_vec[7,2]]
     Coords[global_count+8][:] = [iosx+v_vec[8,0],iosy+v_vec[8,1],iosz+v_vec[8,2]]
     Coords[global_count+9][:] = [iosx+v_vec[9,0],iosy+v_vec[9,1],iosz+v_vec[9,2]]
     Coords[global_count+10][:] = [iosx+v_vec[10,0],iosy+v_vec[10,1],iosz+v_vec[10,2]]
          
     return 


def mer_identity(inp):  
  shiftx = 60
  identities = []
  names = []
  names_gro = []
  SLS_names = ["O1s","C2s","C3s","O4s","H5s","H10s","C6s","H7s","H8s","H9s"]
  SLMD_names = ["O1d","C2d","C3d","O4d","H5d","C6d","H7d","H8d","H9d"]
  SLML_names = ["O1l","C2l","C3l","O4l","H5l","C6l","H7l","H8l","H9l"]
  SLE_names = ["O1e","C2e","C3e","O4e","H5e","O10e","H11e","C6e","H7e","H8e","H9e"]   
  
  SLS_names_gro = ["O1","C2","C3","O4","H5","H10","C6","H7","H8","H9"]
  SLM_names_gro = ["O1","C2","C3","O4","H5","C6","H7","H8","H9"]
  SLE_names_gro = ["O1","C2","C3","O4","H5","O10","H11","C6","H7","H8","H9"]
  for j in range(inp.shape[0]):
   for i in range(shiftx+1,inp.shape[1]-shiftx-3,9):
      if inp[j][i,-1]==1 :
          identities.append(int(0))
          names += SLS_names
          names_gro += SLS_names_gro 
      elif inp[j][i,-2]==1:  
          identities.append(int(1))
          names += SLMD_names
          names_gro += SLM_names_gro
      elif inp[j][i,-3]==1:  
          identities.append(int(2))
          names += SLML_names
          names_gro += SLM_names_gro      
      elif inp[j][i,-4]==1:  
          identities.append(int(3))
          names += SLE_names
          names_gro += SLE_names_gro
  frame_nmer = len(identities)    
  return identities, shiftx, frame_nmer, names, names_gro



def create_chains(frame_idx,epoch,save_path,inp,tar,box):
  print(frame_idx,epoch)

  Pcoords = np.zeros([int(npart),3],dtype=np.float64)
  Tcoords = np.zeros([int(npart),3],dtype=np.float64)
  
  Pst = open(save_path+'PStart_'+str(com_list[frame_idx])+'.gro', 'w')
  Pst.write("PLA\n")
  Pst.write(str(npart)+"\n")
    
  pstd = open(save_path+"PStart_"+str(com_list[frame_idx])+'.dat', 'w')
  pstd.write("PLA\n")
  pstd.write(str(npart)+'\n')
     
  Tst = open(save_path+'TStart_'+str(com_list[frame_idx])+'.gro', 'w')
  Tst.write("PLA\n")
  Tst.write(str(npart)+"\n")
    
  tstd = open(save_path+"TStart_"+str(com_list[frame_idx])+'.dat', 'w')
  tstd.write("PLA\n")
  tstd.write(str(npart)+'\n')     
  identities, shiftx, frame_nmer, frame_names, frame_names_gro = mer_identity(inp) 
  if epoch<10:
      checkpoint_path = "./tmp/cp-000"+ str(epoch)+ ".ckpt"
  elif epoch<100 and epoch>=10:    
      checkpoint_path = "./tmp/cp-00"+ str(epoch)+ ".ckpt"
  elif epoch<1000 and epoch>=100:    
      checkpoint_path = "./tmp/cp-0"+ str(epoch)+ ".ckpt"
  else:    
      checkpoint_path = "./tmp/cp-"+ str(epoch)+ ".ckpt"

  model.built = True  
  model.load_weights(checkpoint_path) 
  pred = model.generator.predict(inp)
  
  pred += 1
  pred /= 2.
  pred *= 0.153001*2
  pred += -0.153001   
  PartIndx = 0  
  c = 0    
  for ii in range(frame_nmer):       

       
        sample_idx = ii//nmer
       
        cg_idx = c+shiftx+1
       
        iosx=float(inp[sample_idx][cg_idx,0])
        iosy=float(inp[sample_idx][cg_idx,1])
        iosz=float(inp[sample_idx][cg_idx,2])
        
        i = shiftx+c

        if identities[ii]==0: 
            spots=spots_SLS  
            pos=np.zeros([spots,3],dtype=np.float64)
            tos=np.zeros([spots,3],dtype=np.float64)
            pos[:] = pred[sample_idx][i:i+spots]
            tos[:] = tar[sample_idx][i:i+spots]
            SLS_d(pos,iosx,iosy,iosz,Pcoords,PartIndx)
            SLS_d(tos,iosx,iosy,iosz,Tcoords,PartIndx)
            PartIndx += merlen_SLS            
        elif identities[ii]==3: 
            spots=spots_SLE  
            pos=np.zeros([spots,3],dtype=np.float64)
            tos=np.zeros([spots,3],dtype=np.float64)
            pos[:] = pred[sample_idx][i:i+spots]
            tos[:] = tar[sample_idx][i:i+spots]
            SLE_d(pos,iosx,iosy,iosz,Pcoords,PartIndx)
            SLE_d(tos,iosx,iosy,iosz,Tcoords,PartIndx)
            PartIndx += merlen_SLE
        elif identities[ii]==1: 
            spots=spots_SLM  
            # print(PartIndx, ii,spots,c) 
            pos=np.zeros([spots,3],dtype=np.float64)
            tos=np.zeros([spots,3],dtype=np.float64)
            pos[:] = pred[sample_idx][i:i+spots]
            tos[:] = tar[sample_idx][i:i+spots]
            SLM_d(pos,iosx,iosy,iosz,Pcoords,PartIndx)
            SLM_d(tos,iosx,iosy,iosz,Tcoords,PartIndx)
            
            PartIndx += merlen_SLM
        elif identities[ii]==2: 
            spots=spots_SLM  
            # print(PartIndx, ii,spots,c) 
            pos=np.zeros([spots,3],dtype=np.float64)
            tos=np.zeros([spots,3],dtype=np.float64)
            pos[:] = pred[sample_idx][i:i+spots]
            tos[:] = tar[sample_idx][i:i+spots]
            SLM_d(pos,iosx,iosy,iosz,Pcoords,PartIndx)
            SLM_d(tos,iosx,iosy,iosz,Tcoords,PartIndx)
            
            PartIndx += merlen_SLM    
        c += spots 

        if PartIndx%chainlen==0:
            c = 0 # initialize for every new sample 
    
        
  for j in range(Pcoords.shape[0]):
         chain_idx = j//(chainlen)
         Pst.write('%5d%5s%5s%5s%8.3f%8.3f%8.3f  0.0000  0.0000  0.0000\n'%((chain_idx+1,'PB',str(frame_names_gro[j]),str(j+1)[-5:],Pcoords[j][0],Pcoords[j][1],Pcoords[j][2]))) 
         print(str(frame_names[j]),Pcoords[j][0],Pcoords[j][1],Pcoords[j][2], file=pstd)   
         Tst.write('%5d%5s%5s%5s%8.3f%8.3f%8.3f  0.0000  0.0000  0.0000\n'%((chain_idx+1,'PB',str(frame_names_gro[j]),str(j+1)[-5:],Tcoords[j][0],Tcoords[j][1],Tcoords[j][2]))) 
         print(str(frame_names[j]),Tcoords[j][0],Tcoords[j][1],Tcoords[j][2], file=tstd)   

  Pst.write('%10.5f%10.5f%10.5f\n'%(( box[0] ,box[1] , box[2] )))
  Pst.close
  pstd.close
  Tst.write('%10.5f%10.5f%10.5f\n'%(( box[0] , box[1] , box[2])))
  Tst.close
  tstd.close
  return   
    
     
if __name__ == '__main__': 
 epochs = [EPOCHS]
 for i in epochs: 
   save_path="./Data"+"_"+str(i)+"/"       
   os.mkdir(save_path)
   pool = multiprocessing.Pool(6)
   start_time = time.perf_counter()
   frames_idxs = [i for i in range(len(com_list))]
   processes = [pool.apply_async(create_chains, args=(x,i,save_path,test_input[x],test_target[x],test_boxes[x],)) for x in frames_idxs] 
   result = [p.get() for p in processes]

   pool.close()  
   finish_time = time.perf_counter()
   print(f"Program finished in {finish_time-start_time} seconds")     

