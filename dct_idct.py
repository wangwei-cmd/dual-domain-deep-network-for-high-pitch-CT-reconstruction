import tensorflow as tf
import numpy as np

def dct1d(x):  ##x.shape=[batch,1968,736,64,1]   dct on dimensions of size 64 and 736
    x=tf.transpose(x,[0,1,2,4,3])   ##shape=[batch,1968,736,1,64]
    xdct=tf.signal.dct(x,norm='ortho') 
    xdct=tf.transpose(xdct,[0,1,2,4,3])  ##shape=[batch,1968,1,64,736]
    return xdct

def idct1d(xdct):  ##shape=[batch,1968,736,64,1] 
    x=tf.transpose(xdct,[0,1,2,4,3])  ##shape=[batch,1968,736,1,64]
    x=tf.signal.idct(x,norm='ortho')  
    x=tf.transpose(x,[0,1,2,4,3])   ##shape=[batch,1968,736,64,1]
    return x



def dct2d(x):  ##x.shape=[batch,1968,736,64,1]   dct on dimensions of size 64 and 736
    x=tf.transpose(x,[0,1,2,4,3])   ##shape=[batch,1968,736,1,64]
    xdct=tf.signal.dct(x,norm='ortho') 
    xdct=tf.transpose(xdct,[0,1,3,4,2])  ##shape=[batch,1968,1,64,736]
    xdct=tf.signal.dct(xdct,norm='ortho')
    xdct=tf.transpose(xdct,[0,1,4,3,2])  ##shape=[batch,1968,736,64,1]
    return xdct

def idct2d(xdct):  ##shape=[batch,1968,736,64,1]
    xdct=tf.transpose(xdct,[0,1,4,3,2])##shape=[batch,1968,1,64,736]
    x=tf.signal.idct(xdct,norm='ortho')  
    x=tf.transpose(x,[0,1,4,2,3])  ##shape=[batch,1968,736,1,64]
    x=tf.signal.idct(x,norm='ortho')  
    x=tf.transpose(x,[0,1,2,4,3])   ##shape=[batch,1968,736,64,1]
    return x

def rfft2d(x):  ##x.shape=[batch,1968,736,64,1]   rfft on dimensions of size 64 and 736
    x=tf.transpose(x,[0,1,4,2,3])   ##shape=[batch,1968,1,736,64]
    # shape=tf.shape(x)
    xdct=tf.signal.rfft2d(x,[736,64]) 
    xdct=tf.transpose(xdct,[0,1,3,4,2])##shape=[batch,1968,736,64,1]
    return xdct

def irfft2d(xdct):  ##shape=[batch,1968,736,64,1]
    xdct=tf.transpose(xdct,[0,1,4,2,3])##shape=[batch,1968,1,736,64]
    x=tf.signal.irfft2d(xdct)  
    x=tf.transpose(x,[0,1,3,4,2],[736,64])  ##shape=[batch,1968,736,64,1]
    return x



if __name__=='__main__':
    # pre='/media/ubuntu/2T/helical_CT_data_fast_pitch/'
    pre='/data/'
    sin_label=np.load(pre+'sin_label_train.npy')
    x=sin_label[0:2]
    x=np.expand_dims(x,-1)
    x=np.transpose(x,[0,3,1,2,4])
    xdct=dct1d(x)
    ix=idct1d(xdct)
    xfft=rfft2d(x)
    ixx=irfft2d(xfft)
    print(tf.reduce_max(abs(ix-x)))
    print(tf.reduce_max(abs(ixx-x)))

    x1=x
    x1[:,:,:,0:20]=0
    x1[:,:,:,44::]=0
    x1dct=dct1d(x1)
    x1fft=rfft2d(x1)
    print('debug')