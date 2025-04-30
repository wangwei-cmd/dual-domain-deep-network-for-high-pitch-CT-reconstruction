import h5py
import numpy as np
import tensorflow as tf
from helical_utis_slice_no_tf_function import helical_rebin_cor_py,helical_no_rebin_cor_backproj_py,\
     parasetupbrindex,parasetup2d_nearest,parasetup1d,helical_backproj_nearest_py
import fast_pitch_unet.unet3d_sin_dct as sinmodel
import fast_pitch_unet.unet2d_ct as ctmodel

pre='./data/rec_p=0.9/merge/'
# pre='/data/'
# pre='/media/ubuntu/8T2/helical_CT_data_fast_pitch/'
# sin_label=np.load(pre+'sin_label_train.npy')
# img_label=np.load(pre+'img_label_train.npy')
# sin_ini=np.load(pre+'sin_ini_train.npy')
# img_ini=np.load(pre+'img_ini_train.npy')
sin_label=np.load(pre+'sin_label_test.npy')
img_label=np.load(pre+'img_label_test.npy')
sin_ini=np.load(pre+'sin_ini_test.npy')
img_ini=np.load(pre+'img_ini_test.npy')

M = np.max(np.max(np.max(img_ini, 1), 1),1)
M=np.reshape(M, [np.shape(M)[0],1,1,1])
# img_label=img_label/M*255
# img_ini=img_ini/M*255
sin_ini=sin_ini/M*255
sin_label=sin_label/M*255

with tf.device('/device:gpu:1'):
    pre1='./data/'
    post1='/64_geometry_1/'
    rebin=h5py.File(pre1+'/py/'+post1+'/helical_back_cycle.mat', 'r')
    alpha_cor, delt_theta = rebin['alpha_cor'][:], rebin['delt_theta'][:][0]
    delt_theta = delt_theta.astype('float32')
    alpha_cor = alpha_cor.astype('float32')
    cos_alpha_cor = np.cos(alpha_cor)
    delt_alpha = alpha_cor[1] - alpha_cor[0]
    DSD = rebin['DSD'][:][0].astype('float32')
    w_phi, dist= rebin['w_phi'][:], rebin['dist'][:]
    w_phi = np.transpose(w_phi, [1, 0]).astype('float32')
    dist = np.transpose(dist, [1, 0]).astype('float32')
    w1,index1 = parasetup1d(w_phi)
    del rebin

    brindex, brweight=np.load(pre1+'/py/'+post1+'/k_index.npy'),\
                        np.load(pre1+'/py/'+post1+'/k_weight.npy')
    brindex, brweight=brindex.astype('float32'), brweight.astype('float32')
    h_matrix = np.load(pre1+'/py/'+post1+'/h_matrix.npy')
    t1,t2=np.load(pre1+'py/'+post1+'/t1.npy'),\
            np.load(pre1+'py/'+post1+'/t2.npy')
    t1,t2=t1[:,0].astype('int32'),t2[:,0].astype('int32')
    brindex = parasetupbrindex(brindex)
    alpha,w,v_star,index_star=\
        np.load(pre1+'py/'+post1+'/alpha_q_1.npy'),\
        np.load(pre1+'py/'+post1+'/w_q_1.npy'),\
        np.load(pre1+'py/'+post1+'/v_s_1.npy'),\
        np.load(pre1+'py/'+post1+'/ind_1.npy')
    index5 =parasetup2d_nearest(alpha, w)
    for i in range(2,t1.shape[0]+1):
        alpha=np.load(pre1+'py/'+post1+'/alpha_q_'+str(i)+'.npy')
        w=np.load(pre1+'py/'+post1+'/w_q_'+str(i)+'.npy')
        v=np.load(pre1+'py/'+post1+'/v_s_'+str(i)+'.npy')
        indt=np.load(pre1+'py/'+post1+'/ind_'+str(i)+'.npy')
        ind5 =parasetup2d_nearest(alpha, w)
        index5=tf.concat([index5,ind5],0)
        v_star=tf.concat([v_star,v],0)
        index_star=tf.concat([index_star,indt],0)
    del alpha,w,v,indt

    # rf=helical_backproj_nearest_py(sin_ini[-2:-1], delt_alpha, delt_theta, w1_24, index1_24, dist_24, DSD,
    #                      h_matrix, brweight, brindex, cos_alpha_cor, 
    #                      index5, v_star, index_star,t1,t2)
    # rf1=helical_backproj_nearest_py(sin_label[-2:-1], delt_alpha, delt_theta, w1, index1, dist, DSD,
    #                      h_matrix, brweight, brindex, cos_alpha_cor, 
    #                      index5, v_star, index_star,t1,t2)

with tf.device('/device:gpu:1'):
    delt_alpha, delt_theta=tf.constant(delt_alpha),tf.constant(delt_theta)
    w1, index1, =tf.constant(w1),tf.constant(index1)
    dist, h_matrix=tf.constant(dist),tf.constant(h_matrix)
    brweight, brindex=tf.constant(brweight), tf.constant(brindex) 
    cos_alpha_cor,index5=tf.constant(cos_alpha_cor),tf.constant(index5)
    v_star, index_star=tf.constant(v_star),tf.constant(index_star)
    t1,t2=tf.constant(t1),tf.constant(t2)
batch=1
sin_model=sinmodel.make_model(batch=batch,shape=(736,24,1968))
ct_model=ctmodel.make_model(shape=(512,512,11))
ckpt_sin = './ckpt/fp_two_model_unet_sin' 
ckpt_ct = './ckpt/fp_two_model_unet_ct'
_=sin_model(sin_ini[0:batch])
_=ct_model(img_ini[0:batch])
sin_model.load_weights(ckpt_sin).expect_partial()
ct_model.load_weights(ckpt_ct).expect_partial()
iter=list(range(0,sin_ini.shape[0],batch))
rf=np.zeros_like(img_label)
pp=np.zeros(sin_ini.shape[0])
qq=np.zeros(sin_ini.shape[0])
for i in range(len(iter)):
    de_sin=sin_model(sin_ini[iter[i]:iter[i]+batch])
    with tf.device('/device:gpu:1'):
        tmp=helical_backproj_nearest_py(de_sin, delt_alpha, delt_theta,
                                    w1, index1, dist, DSD,
                                    h_matrix, brweight, brindex, cos_alpha_cor,
                                    index5, v_star, index_star,t1,t2)
    tmp=ct_model(tmp, training=1)
    rf[iter[i]:iter[i] + batch]=tmp.numpy()/255.0*M[iter[i]:iter[i]+batch]
    # pp[iter[i]:iter[i]+batch]=tf.image.psnr(tmp,
    #  img_label[iter[i]:iter[i]+batch],tf.reduce_max(img_label[iter[i]:iter[i]+batch]))
    # qq[iter[i]:iter[i] + batch] = tf.image.ssim(tmp,img_label[iter[i]:iter[i] + batch],
    #                       tf.reduce_max(img_label[iter[i]:iter[i] + batch]))
shape=np.shape(rf)
rf1=np.zeros([shape[0]*shape[-1],shape[1],shape[2],1])
img1=np.zeros([shape[0]*shape[-1],shape[1],shape[2],1])
img_ini1=np.zeros([shape[0]*shape[-1],shape[1],shape[2],1])
for i in range(shape[0]):
    for j in range(shape[-1]):
        rf1[i*shape[-1]+j,:,:,0]=rf[i,:,:,j]
        img1[i*shape[-1]+j,:,:,0]=img_label[i,:,:,j]
        img_ini1[i*shape[-1]+j,:,:,0]=img_ini[i,:,:,j]
def cal_rmse(y,y1,ux=512,uy=512):
    e=tf.math.square(y-y1)/ux/uy
    rmse=tf.math.sqrt(tf.reduce_sum(e,[1,2,3]))
    return rmse

pp=cal_rmse(rf1,img1)
qq=tf.image.ssim(rf1,img1,tf.reduce_max(img1))
psnr=tf.image.psnr(rf1,img1,tf.reduce_max(img1))
pp1=cal_rmse(img_ini1,img1)
qq1=tf.image.ssim(img_ini1,img1,tf.reduce_max(img1))
psnr1=tf.image.psnr(img_ini1,img1,tf.reduce_max(img1))
print('rmse:',tf.reduce_mean(pp).numpy(),u"\u00B1",
              tf.sqrt(tf.math.reduce_variance(pp)).numpy())
print('psnr:',tf.reduce_mean(psnr).numpy(),u"\u00B1",
              tf.sqrt(tf.math.reduce_variance(psnr)).numpy())
print('ssim:',tf.reduce_mean(qq).numpy(),u"\u00B1",
              tf.sqrt(tf.math.reduce_variance(qq)).numpy())
print('rmse:',tf.reduce_mean(pp1).numpy(),u"\u00B1",
              tf.sqrt(tf.math.reduce_variance(pp1)).numpy())
print('psnr:',tf.reduce_mean(psnr1).numpy(),u"\u00B1",
              tf.sqrt(tf.math.reduce_variance(psnr1)).numpy())
print('ssim:',tf.reduce_mean(qq1).numpy(),u"\u00B1",
              tf.sqrt(tf.math.reduce_variance(qq1)).numpy())
print('debug') 
# np.save('FP_rf_unet3d.npy',rf1)
