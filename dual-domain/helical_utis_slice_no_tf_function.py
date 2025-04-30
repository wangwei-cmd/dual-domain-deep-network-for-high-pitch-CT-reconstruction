import tensorflow as tf
# import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
# from skimage.transform import radon,iradon,rotate
# from scipy.fftpack import fft, ifft, fftfreq, fftshift
import random
import h5py
# from scipy.interpolate import interp2d

def parasetup3d(xq,yq,zq):
    x0 = tf.math.floor(xq)
    y0 = tf.math.floor(yq)
    z0 = tf.math.floor(zq)
    x1 = tf.math.ceil(xq)
    y1 = tf.math.ceil(yq)
    z1 = tf.math.ceil(zq)
    wx1 = xq - x0
    wy1 = yq - y0
    wz1 = zq - z0
    wx0 = 1 - wx1
    wy0 = 1 - wy1
    wz0 = 1 - wz1
    w000 = wx0 * wy0 * wz0
    w100 = wx1 * wy0 * wz0
    w010 = wx0 * wy1 * wz0
    w110 = wx1 * wy1 * wz0
    w001 = wx0 * wy0 * wz1
    w011 = wx0 * wy1 * wz1
    w101 = wx1 * wy0 * wz1
    w111 = wx1 * wy1 * wz1
    index000 = tf.cast(tf.stack([x0, y0, z0], 4),tf.int32)
    index100 = tf.cast(tf.stack([x1, y0, z0], 4),tf.int32)
    index010 = tf.cast(tf.stack([x0, y1, z0], 4),tf.int32)
    index110 = tf.cast(tf.stack([x1, y1, z0], 4),tf.int32)
    index001 = tf.cast(tf.stack([x0, y0, z1], 4),tf.int32)
    index011 = tf.cast(tf.stack([x0, y1, z1], 4),tf.int32)
    index101 = tf.cast(tf.stack([x1, y0, z1], 4),tf.int32)
    index111 = tf.cast(tf.stack([x1, y1, z1], 4),tf.int32)
    return tf.stack([w000, w100, w010, w110, w001, w011, w101, w111]), \
           tf.stack([index000, index100, index010, index110, index001, index011, index101, index111])

# def parasetup3d(xq,yq,zq):
#     x0 = np.floor(xq)
#     y0 = np.floor(yq)
#     z0 = np.floor(zq)
#     x1 = np.ceil(xq)
#     y1 = np.ceil(yq)
#     z1 = np.ceil(zq)
#     wx1 = xq - x0
#     wy1 = yq - y0
#     wz1 = zq - z0
#     wx0 = 1 - wx1
#     wy0 = 1 - wy1
#     wz0 = 1 - wz1
#     w000 = wx0 * wy0 * wz0
#     w100 = wx1 * wy0 * wz0
#     w010 = wx0 * wy1 * wz0
#     w110 = wx1 * wy1 * wz0
#     w001 = wx0 * wy0 * wz1
#     w011 = wx0 * wy1 * wz1
#     w101 = wx1 * wy0 * wz1
#     w111 = wx1 * wy1 * wz1
#     index000 = np.stack([x0, y0, z0], 4).astype('int32')
#     index100 = np.stack([x1, y0, z0], 4).astype('int32')
#     index010 = np.stack([x0, y1, z0], 4).astype('int32')
#     index110 = np.stack([x1, y1, z0], 4).astype('int32')
#     index001 = np.stack([x0, y0, z1], 4).astype('int32')
#     index011 = np.stack([x0, y1, z1], 4).astype('int32')
#     index101 = np.stack([x1, y0, z1], 4).astype('int32')
#     index111 = np.stack([x1, y1, z1], 4).astype('int32')
#     return [w000, w100, w010, w110, w001, w011, w101, w111], \
#            [index000, index100, index010, index110, index001, index011, index101, index111]

def parasetup3d_nearest(xq,yq,zq):
    x0 = tf.math.round(xq)
    y0 = tf.math.round(yq)
    z0 = tf.math.round(zq)
    index000 = tf.cast(tf.stack([x0, y0, z0], 4),tf.int32)
    return index000

# def parasetup3d_nearest(xq,yq,zq):
#     x0 = np.round(xq)
#     y0 = np.round(yq)
#     z0 = np.round(zq)
#     index000 = np.stack([x0, y0, z0], 4).astype('int32')
#     return index000


def parasetup2d(pu, pv):
    u0 = np.floor(pu)
    v0 = np.floor(pv)
    u1 = np.ceil(pu)
    v1 = np.ceil(pv)
    wu1 = pu - u0
    wv1 = pv - v0
    wu0 = 1 - wu1
    wv0 = 1 - wv1
    w00 = wu0 * wv0
    w01 = wu0 * wv1
    w10 = wu1 * wv0
    w11 = wu1 * wv1
    shape=pu.shape
    z = np.arange(shape[0])
    z = np.broadcast_to(z, [shape[1],shape[2],shape[3],shape[0]])
    z=np.transpose(z,[3,0,1,2])
    index00 = np.stack([z, u0, v0], 4).astype('int32')
    index01 = np.stack([z, u0, v1], 4).astype('int32')
    index10 = np.stack([z, u1, v0], 4).astype('int32')
    index11 = np.stack([z, u1, v1], 4).astype('int32')
    return [w00, w01, w10, w11], [index00, index01, index10, index11]

def parasetup2d_nearest(pu, pv):
    u0 = tf.cast(tf.math.round(pu),tf.int32)
    v0 = tf.cast(tf.math.round(pv),tf.int32)
    shape=tf.shape(pu)
    z = tf.range(shape[0])
    z = tf.broadcast_to(z, [shape[1],shape[2],shape[0]])
    z=tf.transpose(z,[2,0,1])
    index = tf.cast(tf.stack([z, u0, v0], 3),tf.int32)
    return index

# def parasetup2d_nearest(pu, pv):
#     u0 = np.round(pu).astype('int32')
#     v0 = np.round(pv).astype('int32')
#     shape=pu.shape
#     z = np.arange(shape[0])
#     z = np.broadcast_to(z, [shape[1],shape[2],shape[0]])
#     z=np.transpose(z,[2,0,1])
#     index = np.stack([z, u0, v0], 3).astype('int32')
#     return index

def parasetup1d(pu,pu_min=0,pu_max=63):
    u0 = tf.math.floor(pu)
    u1 = tf.math.ceil(pu)
    wu1 = pu - u0
    wu0 = 1 - wu1
    z = tf.range(pu.shape[0])
    z = tf.broadcast_to(z, [u0.shape[1],u0.shape[0]])
    z = tf.transpose(z)
    index0 = tf.stack([z,tf.cast(u0,tf.int32)], 2)
    index1 = tf.stack([z,tf.cast(u1,tf.int32)], 2)
    wu0=tf.where(u0<pu_min,tf.zeros_like(u0),wu0)
    wu0=tf.where(u1>pu_max,tf.zeros_like(u0),wu0)
    wu1=tf.where(u0<pu_min,tf.zeros_like(u0),wu1)
    wu1=tf.where(u1>pu_max,tf.zeros_like(u0),wu1)
    return tf.stack([wu0, wu1]), tf.stack([index0,index1])


# def parasetup1d(pu,pu_min=0,pu_max=63):
#     u0 = np.floor(pu)
#     u1 = np.ceil(pu)
#     wu1 = pu - u0
#     wu0 = 1 - wu1
#     z = np.arange(pu.shape[0])
#     z = np.broadcast_to(z, [u0.shape[1],u0.shape[0]])
#     z = np.transpose(z)
#     index0 = np.stack([z,u0], 2).astype('int32')
#     index1 = np.stack([z,u1], 2).astype('int32')
#     wu0[np.where(u0<pu_min)]=0
#     wu0[np.where(u1>pu_max)]=0
#     wu1[np.where(u0<pu_min)]=0
#     wu1[np.where(u1>pu_max)]=0
#     return [wu0, wu1], [index0,index1]

# def parasetupbrindex(index):
#     z = tf.range(index.shape[0])
#     z = tf.broadcast_to(z, [index.shape[1],index.shape[0]])
#     z = tf.transpose(z)
#     index0 = tf.stack([z,tf.cast(index-1,tf.int32)], 2)
#     index1 = tf.stack([z,tf.cast(index,tf.int32)], 2)
#     index2=tf.stack([index0,index1])
#     return index2

def parasetupbrindex(index):
    z = np.arange(index.shape[0])
    z = np.broadcast_to(z, [index.shape[1],index.shape[0]])
    z = np.transpose(z)
    index0 = np.stack([z,index-1], 2).astype('int32')
    index1 = np.stack([z,index], 2).astype('int32')
    return [index0,index1]

# @tf.function
def helical_proj(w,index,img,delt_t):
    img=tf.transpose(img,[1,2,3,0])
    pf=tf.gather_nd(img*1.0,index[0])*tf.expand_dims(w[0],-1)+\
       tf.gather_nd(img*1.0,index[1])*tf.expand_dims(w[1],-1)+\
       tf.gather_nd(img*1.0,index[2])*tf.expand_dims(w[2],-1)+\
       tf.gather_nd(img*1.0,index[3])*tf.expand_dims(w[3],-1)+\
       tf.gather_nd(img*1.0,index[4])*tf.expand_dims(w[4],-1)+\
       tf.gather_nd(img*1.0,index[5])*tf.expand_dims(w[5],-1)+\
       tf.gather_nd(img*1.0,index[6])*tf.expand_dims(w[6],-1)+\
       tf.gather_nd(img*1.0,index[7])*tf.expand_dims(w[7],-1)
    pf=tf.reduce_sum(pf,3)
    pf=tf.transpose(pf,[3,1,2,0])
    return pf*delt_t

def helical_proj_nearest(index,img,delt_t,batch=50):
    img=tf.transpose(img,[1,2,3,0])
    iter=list(range(0,index.shape[3],batch))
    pf=tf.gather_nd(img*1.0,index[:,:,:,iter[0]:iter[0]+batch])
    pf=tf.reduce_sum(pf,3)
    for i in range(1,len(iter)):
        tmp=tf.gather_nd(img*1.0,index[:,:,:,iter[i]:iter[i]+batch])
        tmp=tf.reduce_sum(tmp,3)
        pf=pf+tmp
    pf=tf.transpose(pf,[3,1,2,0])
    return pf*delt_t

@tf.function
def compute_grad_s(pf,delt_alpha,delt_theta):
    print('tracing:compute_grad_s')
    d_proj=(pf[:,:,0:-1,2::]-pf[:,:,0:-1,0:-2])/4/delt_theta+\
           (pf[:,:,1::,2::]-pf[:,:,1::,0:-2])/4/delt_theta
    d_proj=tf.pad(d_proj,[[0,0],[0,0],[0,1],[1,1]],mode='CONSTANT', constant_values=0)
    d_col = (pf[:, 1::]-pf[:, 0:-1])/delt_alpha
    d_col = tf.pad(d_col, [[0, 0], [0,1], [0,0], [0, 0]], mode='CONSTANT', constant_values=0)
    g1=d_proj+d_col
    return g1

# def rebin_cor(w,index,g1,dist,DSD,batch=500):
#     g1=tf.transpose(g1, [1,3,0,2])
#     g1=DSD*g1/dist
#     g1=tf.transpose(g1,[0,3,1,2])
#     iter=list(range(0,g1.shape[2],batch))
#     g2=w[0]*tf.transpose(tf.gather_nd(g1[:,:,
#          iter[0]:iter[0]+batch]*1.0,index[0]),[2,3,0,1])\
#           +w[1]*tf.transpose(tf.gather_nd(g1[:,:,
#          iter[0]:iter[0]+batch]*1.0,index[1]),[2,3,0,1])
#     for i in range(1,len(iter)):
#         tmp=w[0]*tf.transpose(tf.gather_nd(g1[:,:,
#             iter[i]:iter[i]+batch]*1.0,index[0]),[2,3,0,1])\
#            +w[1]*tf.transpose(tf.gather_nd(g1[:,:,
#              iter[i]:iter[i]+batch]*1.0,index[1]),[2,3,0,1])
#         g2=tf.concat([g2,tmp],0)
#     g2=tf.transpose(g2,[1,0,2,3])
#     return g2

@tf.function
def rebin_cor(w,index,g1,dist,DSD):
    print('tracing:rebin_cor')
    g1=tf.transpose(g1, [1,3,0,2])
    g2=DSD*g1/dist
    g2=tf.transpose(g2,[0,3,1,2])
    g3=w[0]*tf.transpose(tf.gather_nd(g2*1.0,index[0]),[2,3,0,1])\
       +w[1]*tf.transpose(tf.gather_nd(g2*1.0,index[1]),[2,3,0,1])
    g3=tf.transpose(g3,[1,0,2,3])
    return g3

@tf.function
def Htransform_cor(g3, h_matrix, delt_alpha):
    print('tracing:Htransform_cor')
    g3 = tf.transpose(g3, [0, 3, 2, 1])
    g3 = tf.matmul(h_matrix, g3)
    g3 = tf.transpose(g3, [0, 3, 2, 1])
    return g3*delt_alpha

@tf.function
def back_rebin_cor(g4,weight,index,cos_alpha_cor):
    print('tracing:back_rebin_cor')
    g4=tf.transpose(g4,[2,3,0,1])
    g4=tf.transpose(tf.gather_nd(g4*1.0,index[0]),[3,2,0,1])*(1-weight)+\
       tf.transpose(tf.gather_nd(g4*1.0,index[1]),[3,2,0,1])*weight
    g4=cos_alpha_cor*g4
    return g4

@tf.function
def back_proj_nearest_py(gc,index,v_star,index_star,t1,t2):
    print('tracing:back_proj_nearest_py')
    gc=tf.transpose(gc,[0,2,3,1])
    gc=tf.gather_nd(gc*1.0,index)
    gc=tf.transpose(gc,[3,0,1,2])
    #### gc=gc/v_star*index_star
    gc=tf.math.multiply(tf.math.divide(gc,v_star),index_star)
    gc=tf.expand_dims(tf.reduce_sum(gc,1),-1)
    return gc

def back_proj_bilinear_py(gc,index,weight,v_star,index_star,t1,t2):
    
    gc=tf.transpose(gc,[0,2,3,1])
    gc=weight[0]*tf.gather_nd(gc*1.0,index[0])+\
       weight[1]*tf.gather_nd(gc*1.0,index[1])+\
       weight[2]*tf.gather_nd(gc*1.0,index[2])+\
       weight[3]*tf.gather_nd(gc*1.0,index[3])
    gc=tf.transpose(gc,[3,0,1,2])
    #### gc=gc/v_star*index_star
    gc=tf.math.multiply(tf.math.divide(gc,v_star),index_star)
    gc=tf.expand_dims(tf.reduce_sum(gc,1),-1)
    return gc

# @tf.function
def helical_backproj_nearest_py(pf, delt_alpha, delt_theta,w1, index1,dist, DSD,
                     h_matrix,brweight, brindex, cos_alpha_cor,
                     index5, v_star, index_star,t1,t2):
    # print('tracing:helical_rec')
    LL=t2-t1
    L=LL[0]
    i=0
    g1 = compute_grad_s(pf[:,:,:,t1[i]:t2[i]], delt_alpha, delt_theta)
    # g3 = rebin_cor(w1, index1, g1[:,:,:,t1[i]:t2[i]], dist, DSD)
    g3 = rebin_cor(w1, index1, g1, dist, DSD)
    g3 = Htransform_cor(g3,h_matrix,delt_alpha)
    g3 = back_rebin_cor(g3, brweight, brindex, cos_alpha_cor)
    rf = back_proj_nearest_py(g3,index5[0:LL[0]],v_star[0:LL[0]],
                        index_star[0:LL[0]],t1,t2)

    for i in range(1,len(t1)):
        g1 = compute_grad_s(pf[:,:,:,t1[i]:t2[i]], delt_alpha, delt_theta)
        g3 = rebin_cor(w1, index1, g1, dist, DSD)
        g3 = Htransform_cor(g3,h_matrix,delt_alpha)
        g3 = back_rebin_cor(g3, brweight, brindex, cos_alpha_cor)
        g3 = back_proj_nearest_py(g3,index5[L:L+LL[i]],v_star[L:L+LL[i]],
                           index_star[L:L+LL[i]],t1,t2)
        rf=tf.concat([rf,g3],-1)
        L=L+LL[i]
    rf=rf/2/np.pi
    return rf

# @tf.function
def helical_no_rebin_cor_backproj_py(pf, delt_alpha, delt_theta,w1, index1,dist, DSD,
                     h_matrix,brweight, brindex, cos_alpha_cor,
                     index5, v_star, index_star,t1,t2):
    #####
    #helical_backproj_nearest_py=helical_rebin_cor_py+helical_no_rebin_cor_backproj_py
    #####
    LL=t2-t1
    L=LL[0]
    i=0
    # g1 = compute_grad_s(pf[:,:,:,t1[i]:t2[i]], delt_alpha, delt_theta)
    # g3 = rebin_cor(w1, index1, g1, dist, DSD)
    # g3 = Htransform_cor(g3,h_matrix,delt_alpha)
    g3 = Htransform_cor(pf[:,t1[i]:t2[i],:,:],h_matrix,delt_alpha)
    g3 = back_rebin_cor(g3, brweight, brindex, cos_alpha_cor)
    rf = back_proj_nearest_py(g3,index5[0:LL[0]],v_star[0:LL[0]],
                        index_star[0:LL[0]],t1,t2)

    for i in range(1,len(t1)):
        # g1 = compute_grad_s(pf[:,:,:,t1[i]:t2[i]], delt_alpha, delt_theta)
        # g3 = rebin_cor(w1, index1, g1, dist, DSD)
        g3 = Htransform_cor(pf[:,t1[i]:t2[i],:,:],h_matrix,delt_alpha)
        g3 = back_rebin_cor(g3, brweight, brindex, cos_alpha_cor)
        g3 = back_proj_nearest_py(g3,index5[L:L+LL[i]],v_star[L:L+LL[i]],
                           index_star[L:L+LL[i]],t1,t2)
        rf=tf.concat([rf,g3],-1)
        L=L+LL[i]
    rf=rf/2/np.pi
    return rf

@tf.function
def helical_rebin_cor_py(pf, delt_alpha, delt_theta,w1, index1,dist, DSD):
    #####
    #helical_backproj_nearest_py=helical_rebin_cor_py+helical_no_rebin_cor_backproj_py
    #####
    print('tracing:helical_rebin_cor_py')
    g1 = compute_grad_s(pf, delt_alpha, delt_theta)
    g3 = rebin_cor(w1, index1, g1, dist, DSD)
    return g3



if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    pre='./data/rec/'
    # pre1='./data/py/sp16/'
    pre1='./data_interval=4/py/64/'
    pre2='./data_interval=4/sp/'
    sintag='sin_64_'
    cttag='img_64_'
    # ctini='img_16_'
    # sinini='sin_16_'
    ctini='img_sp16_'
    sinini='sin_sp16_'
    
    img, rf,pf,pf_sparse=np.load(pre + cttag+'L067' + '.npy').astype('float32'),\
            np.load(pre2+ ctini+'L067' + '.npy'),\
            np.load(pre2+ sintag+'L067' + '.npy'),\
            np.load(pre2+ sinini+'L067' + '.npy')
    img,rf,pf,pf_sparse=img[10:11].astype('float32'),\
                        rf[10:11].astype('float32'),\
                        pf[10:11].astype('float32'),\
                        pf_sparse[10:11].astype('float32')

    delt_t=np.load(pre1+'delt_t.npy')
    xq,yq,zq=np.load(pre1+'xq.npy'),np.load(pre1+'yq.npy'),np.load(pre1+'zq.npy')
    index000=parasetup3d_nearest(xq,yq,zq)
    del xq,yq,zq

    rebin=h5py.File(pre1+'/helical_back_cycle.mat', 'r')
    alpha_cor, delt_theta = rebin['alpha_cor'][:], rebin['delt_theta'][:][0]
    delt_theta = delt_theta.astype('float32')
    alpha_cor = alpha_cor.astype('float32')
    cos_alpha_cor = tf.math.cos(alpha_cor)
    delt_alpha = alpha_cor[1] - alpha_cor[0]
    DSD = rebin['DSD'][:][0].astype('float32')
    w_phi, dist= rebin['w_phi'][:], rebin['dist'][:]
    del rebin
    brindex, brweight=np.load(pre1+'/k_index.npy'),\
                      np.load(pre1+'/k_weight.npy')
    brindex, brweight=brindex.astype('float32'), brweight.astype('float32')
    w_phi = np.transpose(w_phi, [1, 0]).astype('float32')
    dist = np.transpose(dist, [1, 0]).astype('float32')
    h_matrix = np.load(pre1+'/h_matrix.npy')
    t1,t2=np.load(pre1+'/t1.npy'),\
          np.load(pre1+'/t2.npy')
    t1,t2=t1[:,0].astype('int32'),t2[:,0].astype('int32')
    alpha,w,v_star,index_star=\
        np.load(pre1+'/alpha_q_1.npy'),\
        np.load(pre1+'/w_q_1.npy'),\
        np.load(pre1+'/v_s_1.npy'),\
        np.load(pre1+'/ind_1.npy')
    index5 =parasetup2d_nearest(alpha, w)
    for i in range(2,t1.shape[0]+1):
        alpha=np.load(pre1+'/alpha_q_'+str(i)+'.npy')
        w=np.load(pre1+'/w_q_'+str(i)+'.npy')
        v=np.load(pre1+'/v_s_'+str(i)+'.npy')
        indt=np.load(pre1+'/ind_'+str(i)+'.npy')
        ind5 =parasetup2d_nearest(alpha, w)
        index5=np.concatenate([index5,ind5],0)
        v_star=np.concatenate([v_star,v],0)
        index_star=np.concatenate([index_star,indt],0)
    del alpha,w,v,indt

    pw_min=0
    pw_max=63
    w1,index1 = parasetup1d(w_phi,pw_min,pw_max)
    brindex = parasetupbrindex(brindex)
    
    # sin=helical_proj_nearest(index000,img,delt_t,batch=50)

    rf1=helical_backproj_nearest_py(pf, delt_alpha, delt_theta, w1, index1, dist, DSD,
                         h_matrix, brweight, brindex, cos_alpha_cor, 
                         index5, v_star, index_star,t1,t2)
    #####
    #helical_backproj_nearest_py=helical_rebin_cor_py+helical_no_rebin_cor_backproj_py
    #####
    tmp=helical_rebin_cor_py(pf, delt_alpha, delt_theta,w1, index1,dist, DSD)
    rf3=helical_no_rebin_cor_backproj_py(tmp, delt_alpha, delt_theta,w1, index1,dist, DSD,
                     h_matrix,brweight, brindex, cos_alpha_cor,
                     index5, v_star, index_star,t1,t2)
    print(tf.reduce_sum(abs(rf1-rf3)))
    rf2=helical_backproj_nearest_py(pf, delt_alpha, delt_theta, w1, index1, dist, DSD,
                         h_matrix, brweight, brindex, cos_alpha_cor, 
                         index5, v_star, index_star,t1,t2)
    print(tf.reduce_sum(abs(rf1-rf2)))

    rf1=1000*(rf1-0.0192)/0.0192
    img=1000*(img-0.0192)/0.0192
    pp=tf.image.psnr(rf1,img,tf.reduce_max(img)-tf.reduce_min(img))
    qq=tf.image.ssim(rf1,img,tf.reduce_max(img)-tf.reduce_min(img))
    print('psnr:',pp)
    print('ssim:',qq)
    print('debug')




