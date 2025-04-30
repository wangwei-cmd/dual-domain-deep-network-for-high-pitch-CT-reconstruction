import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
# from tensorflow.keras.layers import Conv3D, \
#       Dropout, Conv3DTranspose, LeakyReLU,BatchNormalization,MaxPool3D
from keras import layers
Conv3D=layers.Conv3D
Dropout=layers.Dropout
Conv3DTranspose=layers.Conv3DTranspose
LeakyReLU=layers.LeakyReLU
BatchNormalization=layers.BatchNormalization
MaxPool3D=layers.MaxPool3D

def make_model(shape):
    inputs=tf.keras.Input(shape=shape)
    inputs=tf.expand_dims(inputs,-1)
    # inputs_pad=tf.pad(inputs,[[0,0],[0,0],[0,0],[1,2],[0,0]])

    c1=Conv3D(16,(3,3,1),padding='same')(inputs)
    # c1=BatchNormalization()(c1)
    c1=LeakyReLU()(c1)
    c1=Conv3D(16,(3,3,1),padding='same')(c1)
    # c1=BatchNormalization()(c1)
    c1=LeakyReLU()(c1)
    p1=MaxPool3D((2, 2, 1))(c1)
    
    c2=Conv3D(32,(3,3,1),padding='same')(p1)
    # c2=BatchNormalization()(c2)
    c2=LeakyReLU()(c2)
    c2=Conv3D(32,(3,3,1),padding='same')(c2)
    # c2=BatchNormalization()(c2)
    c2=LeakyReLU()(c2)
    p2=MaxPool3D((2, 2, 1))(c2)
    
    c3=Conv3D(64,(3,3,1),padding='same')(p2)
    # c3=BatchNormalization()(c3)
    c3=LeakyReLU()(c3)
    c3=Conv3D(64,(3,3,1),padding='same')(c3)
    # c3=BatchNormalization()(c3)
    c3=LeakyReLU()(c3)
    p3=MaxPool3D((2, 2, 1))(c3)

    c4=Conv3D(128,(3,3,1),padding='same')(p3)
    # c4=BatchNormalization()(c4)
    c4=LeakyReLU()(c4)
    c4=Conv3D(128,(3,3,1),padding='same')(c4)
    # c4=BatchNormalization()(c4)
    c4=LeakyReLU()(c4)
    p4=MaxPool3D((2, 2, 1))(c4)

    c5=Conv3D(256,(3,3,1),padding='same')(p4)
    c5=Conv3D(128,(3,3,1),padding='same')(c5)

    u6=Conv3DTranspose(128, (3,3,1), strides=(2, 2, 1), padding='same') (c5)
    u6=tf.concat([u6,c4],-1)
    c6=Conv3D(128,(3,3,1),padding='same')(u6)
    # c6=BatchNormalization()(c6)
    c6=LeakyReLU()(c6)
    c6=Conv3D(64,(3,3,1),padding='same')(c6)
    # c6=BatchNormalization()(c6)
    c6=LeakyReLU()(c6)

    u7=Conv3DTranspose(64, (3,3,1), strides=(2, 2, 1), padding='same') (c6)
    u7=tf.concat([u7,c3],-1)
    c7=Conv3D(64,(3,3,1),padding='same')(u7)
    # c7=BatchNormalization()(c7)
    c7=LeakyReLU()(c7)
    c7=Conv3D(32,(3,3,1),padding='same')(c7)
    # c7=BatchNormalization()(c7)
    c7=LeakyReLU()(c7)

    u8=Conv3DTranspose(32, (3,3,1), strides=(2, 2, 1), padding='same') (c7)
    u8=tf.concat([u8,c2],-1)
    c8=Conv3D(32,(3,3,1),padding='same')(u8)
    # c8=BatchNormalization()(c8)
    c8=LeakyReLU()(c8)
    c8=Conv3D(16,(3,3,1),padding='same')(c8)
    # c8=BatchNormalization()(c8)
    c8=LeakyReLU()(c8)

    u9=Conv3DTranspose(16, (3,3,1), strides=(2, 2, 1), padding='same') (c8)
    u9=tf.concat([u9,c1],-1)
    c9=Conv3D(16,(3,3,1),padding='same')(u9)
    # c9=BatchNormalization()(c9)
    c9=LeakyReLU()(c9)
    c9=Conv3D(16,(3,3,1),padding='same')(c9)
    # c9=BatchNormalization()(c9)
    c9=LeakyReLU()(c9)
    c9=Conv3D(1,(3,3,1),padding='same')(c9)
    c9=c9+inputs
    outputs=c9[:,:,:,:,0]
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

def psnr(x, y,max_val=255):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    psnr1=tf.reduce_mean(tf.image.psnr(x, y, max_val=tf.reduce_max(x)))######psnr of f and de_sin
    return [psnr1]

def train(ckpt):
    pre='/data/'
    # pre='/media/ubuntu/2T/helical_CT_data_fast_pitch/'
    # sin_label=np.load(pre+'sin_label_train.npy')
    img_label=np.load(pre+'img_label_train.npy')
    # sin_ini=np.load(pre+'sin_ini_train.npy')
    img_ini=np.load(pre+'img_ini_train.npy')
    M = np.max(np.max(np.max(img_ini, 1), 1),1)
    M=np.reshape(M, [np.shape(M)[0],1,1,1])

    img_label = img_label.astype('float32')
    img_label = img_label/M*255
    img_ini=np.load('./fast_pitch_unet/rf_train.npy').astype('float32')

    
    

    shape=(512,512,11)
    model=make_model(shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[psnr])
    call1=tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)                                
    x=img_ini[0:-4]
    y=img_label[0:-4]
    v_data=(img_ini[-4::],img_label[-4::])
    epoch=500
    hist=model.fit(x,y,batch_size=5,epochs=epoch,validation_data=v_data,
                   callbacks=[call1])
    with open('./trainHistory/unet2d_ct', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ckpt='./ckpt/unet2d_FP_ct'
    train(ckpt)