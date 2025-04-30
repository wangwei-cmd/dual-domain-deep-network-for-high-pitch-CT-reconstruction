import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import h5py
import sys 
import time
sys.path.append(".") 
from helical_utis_slice_no_tf_function import helical_rebin_cor_py,helical_no_rebin_cor_backproj_py,\
     parasetupbrindex,parasetup2d_nearest,parasetup1d,helical_backproj_nearest_py
# from tensorflow.keras.layers import Conv3D, \
#       Dropout, Conv3DTranspose, PReLU,BatchNormalization,MaxPool3D
from keras import layers
Conv3D=layers.Conv3D
Dropout=layers.Dropout
Conv3DTranspose=layers.Conv3DTranspose
PReLU=layers.PReLU
LeakyReLU=layers.LeakyReLU
BatchNormalization=layers.BatchNormalization
MaxPool3D=layers.MaxPool3D
from dct_idct import dct1d,idct1d

def unet(inputs):
    # inputs=tf.keras.Input(shape=shape,batch_size=batch)
    # inputs=tf.expand_dims(inputs,-1)
    # inputs_pad=tf.pad(inputs,[[0,0],[3,4],[0,0],[0,0],[0,0]])
    N=6
    c1=Conv3D(N,(3,1,1),padding='same')(inputs)
    c1=Conv3D(N,(1,3,1),padding='same')(c1)
    c1=Conv3D(N,(1,1,3),padding='same')(c1)
    # c1=BatchNormalization()(c1)
    c1=PReLU(shared_axes=[1,2,3])(c1)
    # c1=Conv3D(1,(3,1,1),padding='same')(c1)
    # c1=Conv3D(1,(1,3,1),padding='same')(c1)
    # c1=Conv3D(1,(1,1,3),padding='same')(c1)

    # # c1=BatchNormalization()(c1)
    # c1=PReLU(shared_axes=[1,2,3])(c1)
    p1=MaxPool3D((2, 2, 2))(c1)
    
    c2=Conv3D(2*N,(3,1,1),padding='same')(p1)
    c2=Conv3D(2*N,(1,3,1),padding='same')(c2)
    c2=Conv3D(2*N,(1,1,3),padding='same')(c2)

    # c2=BatchNormalization()(c2)
    c2=PReLU(shared_axes=[1,2,3])(c2)
    # c2=Conv3D(2,(3,1,1),padding='same')(c2)
    # c2=Conv3D(2,(1,3,1),padding='same')(c2)
    # c2=Conv3D(2,(1,1,3),padding='same')(c2)
    # # c2=BatchNormalization()(c2)
    # c2=PReLU(shared_axes=[1,2,3])(c2)
    p2=MaxPool3D((2, 2, 2))(c2)
    
    c3=Conv3D(4*N,(3,1,1),padding='same')(p2)
    c3=Conv3D(4*N,(1,3,1),padding='same')(c3)
    c3=Conv3D(4*N,(1,1,3),padding='same')(c3)
    # c3=BatchNormalization()(c3)
    # c3=PReLU(shared_axes=[1,2,3])(c3)
    # c3=Conv3D(4,(3,1,1),padding='same')(c3)
    # c3=Conv3D(4,(1,3,1),padding='same')(c3)
    # c3=Conv3D(4,(1,1,3),padding='same')(c3)
    # c3=BatchNormalization()(c3)
    c3=PReLU(shared_axes=[1,2,3])(c3)
    p3=MaxPool3D((2, 2, 2))(c3)

    c4=Conv3D(8*N,(3,1,1),padding='same')(p3)
    c4=Conv3D(8*N,(1,3,1),padding='same')(c4)
    c4=Conv3D(8*N,(1,1,3),padding='same')(c4)
    # c4=BatchNormalization()(c4)
    # c4=PReLU(shared_axes=[1,2,3])(c4)
    # c4=Conv3D(8,(3,1,1),padding='same')(c4)
    # c4=Conv3D(8,(1,3,1),padding='same')(c4)
    # c4=Conv3D(8,(1,1,3),padding='same')(c4)
    # c4=BatchNormalization()(c4)
    c4=PReLU(shared_axes=[1,2,3])(c4)
    p4=MaxPool3D((2, 2, 2))(c4)

    c5=Conv3D(16*N,(3,1,1),padding='same')(p4)
    c5=Conv3D(16*N,(1,3,1),padding='same')(c5)
    c5=Conv3D(16*N,(1,1,3),padding='same')(c5)
    c5=Conv3D(8*N,(3,1,1),padding='same')(c5)
    c5=Conv3D(8*N,(1,3,1),padding='same')(c5)
    c5=Conv3D(8*N,(1,1,3),padding='same')(c5)

    u6=Conv3DTranspose(8*N, (3,1,1), strides=(2, 1, 1), padding='same') (c5)
    u6=Conv3DTranspose(8*N, (1,3,1), strides=(1, 2, 1), padding='same') (u6)
    u6=Conv3DTranspose(8*N, (1,1,3), strides=(1, 1, 2), padding='same') (u6)
    u6=tf.concat([u6,c4],-1)
    # c6=Conv3D(8,(3,1,1),padding='same')(u6)
    # c6=Conv3D(8,(1,3,1),padding='same')(c6)
    # c6=Conv3D(8,(1,1,3),padding='same')(c6)

    # c6=BatchNormalization()(c6)
    c6=PReLU(shared_axes=[1,2,3])(u6)
    c6=Conv3D(4*N,(3,1,1),padding='same')(c6)
    c6=Conv3D(4*N,(1,3,1),padding='same')(c6)
    c6=Conv3D(4*N,(1,1,3),padding='same')(c6)
    # c6=BatchNormalization()(c6)
    c6=PReLU(shared_axes=[1,2,3])(c6)

    u7=Conv3DTranspose(4*N, (3,1,1), strides=(2, 1, 1), padding='same') (c6)
    u7=Conv3DTranspose(4*N, (1,3,1), strides=(1, 2, 1), padding='same') (u7)
    u7=Conv3DTranspose(4*N, (1,1,3), strides=(1, 1, 2), padding='same') (u7)
    u7=tf.concat([u7,c3],-1)
    # c7=Conv3D(4,(3,1,1),padding='same')(u7)
    # c7=Conv3D(4,(1,3,1),padding='same')(c7)
    # c7=Conv3D(4,(1,1,3),padding='same')(c7)
    # c7=BatchNormalization()(c7)
    c7=PReLU(shared_axes=[1,2,3])(u7)
    c7=Conv3D(2*N,(3,1,1),padding='same')(c7)
    c7=Conv3D(2*N,(1,3,1),padding='same')(c7)
    c7=Conv3D(2*N,(1,1,3),padding='same')(c7)
    # c7=BatchNormalization()(c7)
    c7=PReLU(shared_axes=[1,2,3])(c7)

    u8=Conv3DTranspose(2*N, (3,1,1), strides=(2, 1, 1), padding='same') (c7)
    u8=Conv3DTranspose(2*N, (1,3,1), strides=(1, 2, 1), padding='same') (u8)
    u8=Conv3DTranspose(2*N, (1,1,3), strides=(1, 1, 2), padding='same') (u8)
    u8=tf.concat([u8,c2],-1)
    # c8=Conv3D(2,(3,1,1),padding='same')(u8)
    # c8=Conv3D(2,(1,3,1),padding='same')(c8)
    # c8=Conv3D(2,(1,1,3),padding='same')(c8)
    # c8=BatchNormalization()(c8)
    # c8=PReLU(shared_axes=[1,2,3])(c8)
    c8=Conv3D(1*N,(3,1,1),padding='same')(u8)
    c8=Conv3D(1*N,(1,3,1),padding='same')(c8)
    c8=Conv3D(1*N,(1,1,3),padding='same')(c8)
    # c8=BatchNormalization()(c8)
    c8=PReLU(shared_axes=[1,2,3])(c8)

    # u9=Conv3DTranspose(1*N, (1,3,1), strides=(1, 2, 1), padding='same') (c8)
    # u9=Conv3DTranspose(1*N, (1,1,3), strides=(1, 1, 2), padding='same') (u9)
    # u9=Conv3DTranspose(1*N, (3,1,1), strides=(2, 1, 1), padding='same') (u9)
    
    u9=Conv3DTranspose(1*N, (3,1,1), strides=(2, 1, 1), padding='same') (c8)
    u9=Conv3DTranspose(1*N, (1,3,1), strides=(1, 2, 1), padding='same') (u9)
    u9=Conv3DTranspose(1*N, (1,1,3), strides=(1, 1, 2), padding='same') (u9)
    u9=tf.concat([u9,c1],-1)
    # c9=Conv3D(1,(3,1,1),padding='same')(u9)
    # c9=Conv3D(1,(1,3,1),padding='same')(c9)
    # c9=Conv3D(1,(1,1,3),padding='same')(c9)
    # c9=BatchNormalization()(c9)
    # c9=PReLU(shared_axes=[1,2,3])(c9)
    c9=Conv3D(1*N,(3,1,1),padding='same')(u9)
    c9=Conv3D(1*N,(1,3,1),padding='same')(c9)
    c9=Conv3D(1*N,(1,1,3),padding='same')(c9)
    # c9=BatchNormalization()(c9)
    c9=PReLU(shared_axes=[1,2,3])(c9)
    c9=Conv3D(1,(3,1,1),padding='same')(c9)
    c9=Conv3D(1,(1,3,1),padding='same')(c9)
    c9=Conv3D(1,(1,1,3),padding='same')(c9)
    c9=c9+inputs
    # model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return c9

def make_model(batch,shape):
    inputs = tf.keras.Input(shape=shape,batch_size=batch)
    inputs_pad=tf.concat([tf.zeros([batch,shape[0], 20, shape[2]]),\
                                   inputs,\
                                   tf.zeros([batch,shape[0], 20, shape[2]])], axis=2)
    inputs_pad=tf.transpose(inputs_pad,[0,3,1,2])
    inputs_pad=tf.expand_dims(inputs_pad,-1)
    inputs_dct=dct1d(inputs_pad)
    outputs=unet(inputs_dct)
    outputs=idct1d(outputs)
    outputs=tf.transpose(outputs,[0,2,3,1,4])
    outputs=tf.concat([outputs[:,:,0:20,:,0],inputs,
                                   outputs[:,:, -20::, :,0]], axis=2)
    # outputs=tf.concat([outputs[:,:,0:16,:,0],inputs,
    #                                outputs[:,:, -16::, :,0]], axis=2)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)
    return model



@tf.function
def train_step(inputs, model, labels, Loss, Metric, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=1)
        loss = Loss(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # m1 = Metric(labels[:,:,20:44], inputs)
    # m2 = Metric(labels, model(inputs, training=0))
    m2 = Metric(labels, predictions)
    # m3 = Metric(vy, model(vx, training=0))
    return loss, m2

def psnr(x, y,max_val=255):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    psnr1=tf.reduce_mean(tf.image.psnr(x, y, max_val=tf.reduce_max(x)))######psnr of f and de_sin
    return [psnr1]

def train(ckpt):
    pre='/data/'
    # pre='/media/ubuntu/2T/helical_CT_data_fast_pitch/'
    sin_label=np.load(pre+'sin_label_train.npy')
    img_label=np.load(pre+'img_label_train.npy')
    sin_ini=np.load(pre+'sin_ini_train.npy')
    img_ini=np.load(pre+'img_ini_train.npy')

    M = np.max(np.max(np.max(img_ini, 1), 1),1)
    M=np.reshape(M, [np.shape(M)[0],1,1,1])
    img_label=img_label/M*255
    img_ini=img_ini/M*255
    sin_ini=sin_ini/M*255
    sin_label=sin_label/M*255


    # tmp1=unet(np.transpose(sin_label[0:1],[0,3,1,2]))
    batch=1
    shape=(736,24,1968)
    model=make_model(batch,shape)
    # tmp=model(np.transpose(sin_label[0:1],[0,3,1,2]))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    x=sin_ini[0:-4]
    y=sin_label[0:-4]
    v_data=(sin_ini[-4::],sin_label[-4::])
    epoch=100
    batch=1
    N=len(y)
    id=np.array(range(N))
    iter=list(range(0,N,batch))
    tt=np.zeros([N])
    m=0.0
    # model(sin_ini[0:batch])
    
    loss_1=tf.keras.losses.MeanSquaredError()
    # model.compile(optimizer=optimizer,loss=loss_1,
    #              metrics=[psnr])
    for i in range(epoch):
        np.random.shuffle(id)
        for j in range(len(iter)):
            
            f, f_noisy =y[id[iter[j]:iter[j]+batch]],\
                            x[id[iter[j]:iter[j]+batch]]
            f,f_noisy=tf.constant(f),tf.constant(f_noisy)
            tt1=time.process_time()
            model(f_noisy)
            tt2=time.process_time()
            Loss, m2 = train_step(f_noisy, model, f, loss_1, psnr, optimizer)
            print(tt2-tt1,time.process_time()-tt2)
            tt[iter[j]:iter[j]+batch]=m2[0].numpy()
            print(iter[j], "/", i, ":", Loss.numpy(),
                #   "psnr_noisy:", m1[0].numpy(),
                  "psnr_denoi", m2[0].numpy(),
                #   'psnr3:', [m3[0].numpy(), m3[1].numpy(), m3[2].numpy()]
                  )
        # m=0.0      
        if  np.mean(tt)>m:
            m=np.mean(tt)
            model.save_weights(ckpt)

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ckpt='./ckpt/FP_sp_unet3d_dct_sin'
    train(ckpt)




    


