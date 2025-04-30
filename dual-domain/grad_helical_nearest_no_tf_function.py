import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py
from helical_utis_slice_no_tf_function import \
helical_backproj_nearest_py,\
parasetup2d_nearest,parasetup1d,parasetupbrindex

@tf.function
def grad_slice(dy,x,t1,t2,shape):
    print('tracing:grad_slice')
    # shape=tf.shape(x)
    grad=dy*tf.ones([shape[0],shape[1],shape[2],t2-t1],dtype=dy.dtype)
    grad=tf.pad(grad,[[0,0],[0,0],[0,0],[t1,shape[-1]-t2]])
    return grad

@tf.function
def grad_back_proj_nearest_py(dy,gc,index,v_star,index_star,shape):
    print('tracing:grad_back_proj_nearest_py')
    shape=[shape[0],shape[2],shape[3],shape[1]]
    dy=tf.squeeze(dy, -1)
    dy=tf.expand_dims(dy, 1)
    dy=tf.math.multiply(tf.math.divide(dy,v_star),index_star)
    dy=tf.transpose(dy,[1,2,3,0])
    dy=tf.scatter_nd(index,dy,shape)
    dy=tf.transpose(dy,[0,3,1,2])
    return dy

@tf.function
def grad_back_rebin_cor(dy,g4,weight,index,cos_alpha_cor,shape):
    print('tracing:grad_back_rebin_cor')
    shape=[shape[2],shape[3],shape[0],shape[1]]

    dy=cos_alpha_cor*dy
    dy_1=tf.transpose(dy*(1-weight),[2,3,1,0])
    dy_2=tf.transpose(dy*weight,[2,3,1,0])
    dy=tf.scatter_nd(index[0],dy_1,shape)\
       +tf.scatter_nd(index[1],dy_2,shape)
    dy=tf.transpose(dy, [2,3,0,1])
    return dy

@tf.function
def grad_Htransform_cor(dy,g3, h_matrix, delt_alpha):
    print('tracing:grad_Htransform_cor')
    dy=delt_alpha*tf.transpose(dy,[0,3,2,1])
    dy = tf.matmul(h_matrix, dy, adjoint_a=True)
    dy=tf.transpose(dy, [0,3,2,1])
    return dy

@tf.function
def grad_rebin_cor(dy,w,index,g1,dist,DSD):
    print('tracing:grad_rebin_cor')
    dy_shape=tf.shape(dy)
    dist_shape=tf.shape(dist)
    shape=[dy_shape[2],dist_shape[1],dy_shape[1],dy_shape[0]]
    dy=tf.transpose(dy,[1,0,2,3])
    grad=tf.scatter_nd(index[0],tf.transpose(w[0]*dy,[2,3,0,1]),shape)
    grad=grad+tf.scatter_nd(index[1],tf.transpose(w[1]*dy,[2,3,0,1]),shape)
    grad=tf.transpose(grad,[0,2,3,1])
    grad=DSD*grad/dist
    grad=tf.transpose(grad, [2,0,3,1])
    return grad

@tf.function
def grad_compute_grad_s(dy,pf,delt_alpha,delt_theta):
    print('tracing:grad_compute_grad_s')
    cen=dy[:,:,0:-1,1:-3]-dy[:,:,0:-1,3:-1]
    t1=-dy[:,:,0:-1,1:3]
    t2=dy[:,:,0:-1,-3:-1]
    grad=tf.concat([t1,cen,t2],-1)
    grad1=tf.pad(grad, [[0,0],[0,0],[0,1],[0,0]],
                       mode='CONSTANT', constant_values=0)
    grad2=tf.pad(grad, [[0,0],[0,0],[1,0],[0,0]],
                       mode='CONSTANT', constant_values=0)
    grad_proj=(grad1+grad2)/4/delt_theta
    t1=tf.pad(dy[:,0:-1], [[0,0],[1,0],[0,0],[0,0]],
                       mode='CONSTANT', constant_values=0)
    t2=tf.pad(dy[:,0:-1], [[0,0],[0,1],[0,0],[0,0]],
                        mode='CONSTANT', constant_values=0)
    grad_col=(t1-t2)/delt_alpha
    return grad_proj+grad_col

# @tf.function
def grad_helical_backproj_nearest_py(dy,x,delt_alpha, delt_theta, w1, index1, dist, DSD,
                         h_matrix, brweight, brindex, cos_alpha_cor,
                         index5, v_star, index_star,t1,t2):
    
    dtype=dy.dtype
    delt_alpha, delt_theta=tf.cast(delt_alpha,dtype),\
                           tf.cast(delt_theta,dtype)
    # w1=tf.cast(w1,dtype=dtype)
    # dist, DSD,h_matrix=tf.cast(dist,dtype=dtype),\
    #                    tf.cast(DSD,dtype=dtype),\
    #                   tf.cast(h_matrix,dtype=dtype)
    # brweight, cos_alpha_cor=tf.cast(brweight,dtype=dtype),\
    #                         tf.cast(cos_alpha_cor,dtype=dtype)
    # v_star, index_star=tf.cast(v_star,dtype=dtype),\
    #                   tf.cast(index_star,dtype=dtype)
    # print('tracing:grad_helical')
    dy=dy/2/np.pi
    LL=t2-t1
    L=0
    batch=tf.shape(dy)[0]
    shape=tf.shape(brweight)
    pf_shape=[batch,shape[0],shape[1],t2[-1]]
    grad1=tf.zeros(pf_shape,dtype=dy.dtype)
    for i in range(len(t1)):
        shape1=[t2[i]-t1[i],batch,shape[0],shape[1]]
        shape2=[batch,t2[i]-t1[i],shape[0],tf.shape(w1)[2]]
        grad=grad_back_proj_nearest_py(dy[:,:,:,i:i+1],0,
                        index5[L:L+LL[i]],v_star[L:L+LL[i]],
                        index_star[L:L+LL[i]],shape1)
        grad=grad_back_rebin_cor(grad,0,brweight, brindex,
                            cos_alpha_cor,shape2)
        grad=grad_Htransform_cor(grad,0, h_matrix, delt_alpha)
        grad=grad_rebin_cor(grad,w1, index1, 0,dist,DSD)
        grad=grad_compute_grad_s(grad,0,delt_alpha,delt_theta)
        grad1=grad1+grad_slice(grad,0,t1[i],t2[i],pf_shape)
        L=L+LL[i]
    return grad1

def grad_helical_no_rebin_cor_backproj_nearest_py(dy,x,delt_alpha, delt_theta, w1, index1, dist, DSD,
                         h_matrix, brweight, brindex, cos_alpha_cor,
                         index5, v_star, index_star,t1,t2):
    
    dtype=dy.dtype
    delt_alpha, delt_theta=tf.cast(delt_alpha,dtype),\
                           tf.cast(delt_theta,dtype)
    # print('tracing:grad_helical')
    dy=dy/2/np.pi
    LL=t2-t1
    L=0
    batch=tf.shape(dy)[0]
    shape=tf.shape(brweight)
    i=0
    shape1=[t2[i]-t1[i],batch,shape[0],shape[1]]
    shape2=[batch,t2[i]-t1[i],shape[0],tf.shape(w1)[2]]
    grad1=grad_back_proj_nearest_py(dy[:,:,:,i:i+1],0,
                    index5[L:L+LL[i]],v_star[L:L+LL[i]],
                    index_star[L:L+LL[i]],shape1)
    grad1=grad_back_rebin_cor(grad1,0,brweight, brindex,
                        cos_alpha_cor,shape2)
    grad1=grad_Htransform_cor(grad1,0, h_matrix, delt_alpha)
    grad1=tf.pad(grad1,[[0,0],[t1[i],t2[-1]-t2[i]],[0,0],[0,0]])
    L=L+LL[i]
    for i in range(1,len(t1)):
        shape1=[t2[i]-t1[i],batch,shape[0],shape[1]]
        shape2=[batch,t2[i]-t1[i],shape[0],tf.shape(w1)[2]]
        grad=grad_back_proj_nearest_py(dy[:,:,:,i:i+1],0,
                        index5[L:L+LL[i]],v_star[L:L+LL[i]],
                        index_star[L:L+LL[i]],shape1)
        grad=grad_back_rebin_cor(grad,0,brweight, brindex,
                            cos_alpha_cor,shape2)
        grad=grad_Htransform_cor(grad,0, h_matrix, delt_alpha)
        # grad1=tf.concat([grad1,grad],1)
        # grad=grad_rebin_cor(grad,w1, index1, 0,dist,DSD)
        # grad=grad_compute_grad_s(grad,0,delt_alpha,delt_theta)
        grad1=grad1+tf.pad(grad,[[0,0],[t1[i],t2[-1]-t2[i]],[0,0],[0,0]])
        L=L+LL[i]
    return grad1

@tf.function
def grad_helical_rebin_cor_nearest_py(dy,x,delt_alpha, delt_theta, w1, index1, dist, DSD,
                         h_matrix, brweight, brindex, cos_alpha_cor,
                         index5, v_star, index_star,t1,t2):
    print('tracing:grad_helical_rebin_cor_nearest_py')
    dtype=dy.dtype
    delt_alpha, delt_theta=tf.cast(delt_alpha,dtype),\
                           tf.cast(delt_theta,dtype)
    grad1=grad_rebin_cor(dy,w1, index1, 0,dist,DSD)
    grad1=grad_compute_grad_s(grad1,0,delt_alpha,delt_theta)
    return grad1



if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dtype=tf.float64
    # dtype=tf.float32

    if dtype==tf.float32:
        type='float32'
    elif dtype==tf.float64:
        type='float64'
    pre='./128x128/data/'
    pre1=pre+'py/64/'
    x=np.load(pre+'sin_64_L067.npy')
    x=tf.cast(x,dtype)
    y=np.load(pre+'img_64_L067.npy')
    y=tf.cast(y,dtype)

    rebin=h5py.File(pre1+'/helical_back_cycle.mat', 'r')
    alpha_cor, delt_theta = rebin['alpha_cor'][:], rebin['delt_theta'][:][0]
    delt_theta = delt_theta.astype(type)
    alpha_cor = alpha_cor.astype(type)
    cos_alpha_cor = tf.math.cos(alpha_cor)
    delt_alpha = alpha_cor[1] - alpha_cor[0]
    DSD = rebin['DSD'][:][0]
    DSD=tf.cast(DSD,dtype)
    w_phi, dist= rebin['w_phi'][:], rebin['dist'][:]
    del rebin
    brindex, brweight=np.load(pre1+'/k_index.npy'),\
                        np.load(pre1+'/k_weight.npy')
    brindex, brweight=brindex.astype(type),brweight.astype(type)
    w_phi = np.transpose(w_phi, [1, 0]).astype(type)
    dist = np.transpose(dist, [1, 0]).astype(type)
    h_matrix = np.load(pre1+'/h_matrix.npy').astype(type)
    t1,t2=np.load(pre1+'/t1.npy'),\
            np.load(pre1+'/t2.npy')
    t1,t2=t1[:,0].astype('int32'),t2[:,0].astype('int32')
    alpha,w,v_star,index_star=\
        np.load(pre1+'/alpha_q_1.npy').astype(type),\
        np.load(pre1+'/w_q_1.npy').astype(type),\
        np.load(pre1+'/v_s_1.npy').astype(type),\
        np.load(pre1+'/ind_1.npy').astype(type)
    index5 =parasetup2d_nearest(alpha, w)
    for i in range(2,t1.shape[0]+1):
        alpha=np.load(pre1+'/alpha_q_'+str(i)+'.npy').astype(type)
        w=np.load(pre1+'/w_q_'+str(i)+'.npy').astype(type)
        v=np.load(pre1+'/v_s_'+str(i)+'.npy').astype(type)
        indt=np.load(pre1+'/ind_'+str(i)+'.npy').astype(type)
        ind5 =parasetup2d_nearest(alpha, w)
        index5=np.concatenate([index5,ind5],0)
        v_star=np.concatenate([v_star,v],0)
        index_star=np.concatenate([index_star,indt],0)
    del alpha,w,v,indt

    pw_min=0
    pw_max=63
    w1,index1 = parasetup1d(w_phi,pw_min,pw_max)
    brindex = parasetupbrindex(brindex)
    LL=t2-t1
    L=LL[0]
    i=0
    delt_alpha, delt_theta=tf.constant(delt_alpha),tf.constant(delt_theta)
    w1, index1, =tf.constant(w1),tf.constant(index1)
    dist, h_matrix=tf.constant(dist),tf.constant(h_matrix)
    brweight, brindex=tf.constant(brweight), tf.constant(brindex) 
    cos_alpha_cor,index5=tf.constant(cos_alpha_cor),tf.constant(index5)
    v_star, index_star=tf.constant(v_star),tf.constant(index_star)
    t1,t2=tf.constant(t1),tf.constant(t2)
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(x)
        y1=helical_backproj_nearest_py(x,delt_alpha, delt_theta, w1, index1, dist, DSD,
                            h_matrix, brweight, brindex, cos_alpha_cor,
                            index5, v_star, index_star,t1,t2)
        tape1.watch(y1)
        loss=tf.reduce_sum(0.5*tf.math.square(tf.random.normal(y1.shape,dtype=dtype)-y1))
    g_y1,g_x=tape1.gradient(loss,[y1,x])
    G_x=grad_helical_backproj_nearest_py(g_y1,0,delt_alpha, delt_theta, w1, index1, dist, DSD,
                            h_matrix, brweight, brindex, cos_alpha_cor,
                            index5, v_star, index_star,t1,t2)
    tmp=grad_helical_no_rebin_cor_backproj_nearest_py(g_y1,0,delt_alpha, delt_theta, w1, index1, dist, DSD,
                         h_matrix, brweight, brindex, cos_alpha_cor,
                         index5, v_star, index_star,t1,t2)
    G_x1=grad_helical_rebin_cor_nearest_py(tmp,x,delt_alpha, delt_theta, w1, index1, dist, DSD,
                         h_matrix, brweight, brindex, cos_alpha_cor,
                         index5, v_star, index_star,t1,t2)
    print(tf.reduce_sum(abs(G_x-G_x1)))
    print(tf.reduce_sum(abs(G_x-g_x)))
    print('debug')

