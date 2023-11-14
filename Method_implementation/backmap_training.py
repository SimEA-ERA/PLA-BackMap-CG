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

#Load training and validation sets

PATH = "."

with open(PATH+'/train_target.pkl','rb') as f:
    train_target = pickle.load(f)
    print(train_target.shape)

with open(PATH+'/train_input.pkl','rb') as f:
    train_input = pickle.load(f)
    print(train_input.shape)

with open(PATH+'/val_target.pkl','rb') as f:
    val_target = pickle.load(f)
    print(val_target.shape)

with open(PATH+'/val_input.pkl','rb') as f:
    val_input = pickle.load(f)
    print(val_input.shape)

#Normalize the target output in the interval [-1,1]

train_target -= -0.153001
train_target /= 0.153001*2
train_target *= 2
train_target -= 1


val_target -= -0.153001
val_target /= 0.153001*2
val_target *= 2
val_target -= 1

# Create the functions needed for the Loss function 
    
with open('../Data_loss/chain_ba_list.pkl','rb') as f:
    chain_ba_list = pickle.load(f)
    print(len(chain_ba_list))

with open('../Data_loss/chain_ba_sign_list.pkl','rb') as f:
    chain_ba_sign_list = pickle.load(f)
    print(len(chain_ba_sign_list))

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


chain_ba_sign_array = np.array(chain_ba_sign_list)
chain_ba_sign_list_1 = chain_ba_sign_array[:,0]
chain_ba_sign_list_1=tf.cast(chain_ba_sign_list_1,dtype="double")
chain_ba_sign_list_2 = chain_ba_sign_array[:,1]
chain_ba_sign_list_2=tf.cast(chain_ba_sign_list_2,dtype="double")


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


def bond_angles_loss(y_p,y_t,chain_ba_list_1,chain_ba_list_2,chain_ba_sign_list_1,chain_ba_sign_list_2):
    p_chain_ba_list_1 = tf.gather(y_p,chain_ba_list_1,axis=1)
    p_chain_ba_list_2 = tf.gather(y_p,chain_ba_list_2,axis=1)

    p_chain_ba_list_1 = tf.math.multiply(p_chain_ba_list_1 ,tf.cast(chain_ba_sign_list_1[tf.newaxis,:,tf.newaxis],dtype=tf.float32))
    p_chain_ba_list_2 = tf.math.multiply(p_chain_ba_list_2 ,tf.cast(chain_ba_sign_list_2[tf.newaxis,:,tf.newaxis],dtype=tf.float32))

    
    t_chain_ba_list_1 = tf.gather(y_t,chain_ba_list_1,axis=1)
    t_chain_ba_list_2 = tf.gather(y_t,chain_ba_list_2,axis=1)
 
    t_chain_ba_list_1 = tf.math.multiply(t_chain_ba_list_1 ,tf.cast(chain_ba_sign_list_1[tf.newaxis,:,tf.newaxis],dtype=tf.float32))
    t_chain_ba_list_2 = tf.math.multiply(t_chain_ba_list_2 ,tf.cast(chain_ba_sign_list_2[tf.newaxis,:,tf.newaxis],dtype=tf.float32))

     
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


#Define the custom fit function
class u_net(tf.keras.Model):
    def __init__(self, generator ,chain_ba_list_1=chain_ba_list_1 ,chain_ba_list_2=chain_ba_list_2,chain_ba_sign_list_1=chain_ba_sign_list_1 ,chain_ba_sign_list_2=chain_ba_sign_list_2, chain_da_list_1=chain_da_list_1 ,chain_da_list_2=chain_da_list_2,chain_da_list_3=chain_da_list_3,chain_da_sign_list_1=chain_da_sign_list_1 ,chain_da_sign_list_2=chain_da_sign_list_2,chain_da_sign_list_3=chain_da_sign_list_3,**kwargs):
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
            ba_loss = bond_angles_loss(y_pred, y_true,chain_ba_list_1,chain_ba_list_2,chain_ba_sign_list_1,chain_ba_sign_list_2)
            da_loss = dihedral_angles_loss(y_pred, y_true,chain_da_list_1,chain_da_list_2,chain_da_list_3,chain_da_sign_list_1,chain_da_sign_list_2,chain_da_sign_list_3)
          
            total_loss = lbv*reconstruction_loss + lbl*bl_loss + lba*ba_loss + lda*da_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.bl_loss_tracker.update_state(bl_loss)
        self.ba_loss_tracker.update_state(ba_loss)
        self.da_loss_tracker.update_state(da_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.reconstruction_loss_tracker.result(),
            "bl_loss": self.bl_loss_tracker.result(),
            "ba_loss": self.ba_loss_tracker.result(),
            "da_loss": self.da_loss_tracker.result(),
        
        }

    def test_step(self, input_data):
      val_x, val_true = input_data # <-- Seperate X and y
      val_pred = self.generator(val_x)
      val_reconstruction_loss = tf.keras.losses.MeanAbsoluteError()(val_pred, val_true)
      val_bl_loss = bond_lengths_loss(val_pred, val_true)
      val_ba_loss = bond_angles_loss(val_pred, val_true,chain_ba_list_1,chain_ba_list_2,chain_ba_sign_list_1,chain_ba_sign_list_2)
      val_da_loss = dihedral_angles_loss(val_pred, val_true,chain_da_list_1,chain_da_list_2,chain_da_list_3,chain_da_sign_list_1,chain_da_sign_list_2,chain_da_sign_list_3)
      val_total_loss = lbv*val_reconstruction_loss + lbl*val_bl_loss + lba*val_ba_loss + lda*val_da_loss 
      return {"loss": val_total_loss,
              "rec_loss":val_reconstruction_loss,
              "bl_loss": val_bl_loss,
              "ba_loss": val_ba_loss,
              "da_loss": val_da_loss,
              }
        

# Define the callbacks of the model
checkpoint_filepath = './Best_model_weights.h5'
# Model weights are saved at the end of every epoch, if it's the best seen so far.
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss', verbose=1,
    mode='min',
    save_best_only=True)

checkpoint_path = "./tmp/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
   filepath = checkpoint_path, monitor = 'val_loss',
   verbose=1, save_weights_only=True, mode = "min", 
   save_freq='epoch', save_best_only=False)


learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1, factor=0.8, min_lr=0.000001)

# Compile the model
model = u_net(Generator)
model.compile(optimizer=tf.keras.optimizers.Adam())

# Train the model
history = model.fit(train_input,train_target,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,verbose=1,
    shuffle=True,validation_data=(val_input,val_target),
    callbacks=[model_checkpoint_callback,model_checkpoint_callback2,learning_rate_reduction])

#Plot the loss function as a function of epochs 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("loss_plot.png")

#Save the loss function 
with open('losses.pkl','wb') as f:
      pickle.dump(history.history, f)
      
      
      
      
      

