import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import h5py
from helical_utis_slice_no_tf_function import helical_rebin_cor_py,helical_no_rebin_cor_backproj_py,\
     parasetupbrindex,parasetup2d_nearest,parasetup1d,helical_backproj_nearest_py
from  grad_helical_nearest_no_tf_function import grad_helical_rebin_cor_nearest_py,\
grad_helical_no_rebin_cor_backproj_nearest_py, grad_helical_backproj_nearest_py
# import fast_pitch_unet.red_cnn_sin as sinmodel
import fast_pitch_unet.unet3d_sin_dct as sinmodel
import fast_pitch_unet.unet2d_ct as ctmodel
import glob
import time

@tf.function
def train_ct(rf,ct_model,labels,optimizer):
    with tf.GradientTape() as tape:
        tape.watch(rf)
        ct = ct_model(rf, training=1)
        # ct = tf.squeeze(ct,-1)
        # loss = Loss(labels[0], ct)
        loss=tf.keras.losses.MeanSquaredError()(labels[0], ct)
    vars=[rf]+ct_model.trainable_variables
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads[1::], ct_model.trainable_variables))
    return grads[0],ct,loss

@tf.function
def train_sin(grads,labels,sin_model,inputs,ct,optimizer,Metric):
    sshpe=tf.shape(labels[1])
    sshpe=tf.cast(sshpe,tf.float32)
    # fake_label=grads*sshpe[0]*sshpe[1]*sshpe[2]*sshpe[3]+de_sin
    fake_label=grads*sshpe[0]*sshpe[1]*sshpe[2]*sshpe[3]

    with tf.GradientTape() as tape:
        de_sin = sin_model(inputs, training=1)
        # loss = Loss(labels[1]+fake_label, de_sin)
        fake_true=labels[1]+100*fake_label
        loss=tf.keras.losses.MeanSquaredError()(fake_true, de_sin)
    grads = tape.gradient(loss, sin_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, sin_model.trainable_variables))
    m2 = Metric(labels, [ct,de_sin])
    # m3 = Metric(vy, model(vx, training=True))
    return loss, m2

# @tf.function
def train_step(inputs, sin_model,ct_model,
               labels, Loss, Metric, optimizer,
                    delt_alpha, delt_theta,
                    w1, index1, dist, DSD,
                    h_matrix, brweight, brindex, 
                    cos_alpha_cor,index5, v_star, 
                    index_star,t1,t2):
    tt1=time.process_time()
    with tf.device('/device:gpu:0'):
        de_sin=sin_model(inputs, training=1)
    tt2=time.process_time()
    with tf.device('/device:gpu:1'):
        rf=helical_backproj_nearest_py(de_sin, delt_alpha, delt_theta,w1, index1,dist, DSD,
                     h_matrix,brweight, brindex, cos_alpha_cor,
                     index5, v_star, index_star,t1,t2)
    tt3=time.process_time()
    with tf.device('/device:gpu:0'):
        grads,ct,ct_loss=train_ct(rf,ct_model,labels,optimizer)
        print('max_grad_rf:',tf.reduce_max(grads))
    tt4=time.process_time()
    with tf.device('/device:gpu:1'):
        grads=grad_helical_backproj_nearest_py(grads,0,delt_alpha, delt_theta, w1, index1, dist, DSD,
                         h_matrix, brweight, brindex, cos_alpha_cor,
                         index5, v_star, index_star,t1,t2)
    tt5=time.process_time()
    with tf.device('/device:gpu:0'):
        sin_loss,m2=train_sin(grads,labels,sin_model,inputs,ct,optimizer,Metric)
    tt6=time.process_time()
    print(tt2-tt1,tt3-tt2,tt4-tt3,tt5-tt4,tt6-tt5)
    return sin_loss,m2,ct_loss


def train(batch,epoch):
    pre='/data/'
    # pre='/media/ubuntu/2T/helical_CT_data_fast_pitch/'
    sin_label=np.load(pre+'sin_label_train.npy')
    img_label=np.load(pre+'img_label_train.npy')
    sin_ini=np.load(pre+'sin_ini_train.npy')
    img_ini=np.load(pre+'img_ini_train.npy')

    # sin_label = sin_label.astype('float32')
    # img_label = img_label.astype('float32')
    # sin_ini = sin_ini.astype('float32')
    # img_ini = img_ini.astype('float32')

    

    M = np.max(np.max(np.max(img_ini, 1), 1),1)
    M=np.reshape(M, [np.shape(M)[0],1,1,1])
    img_label=img_label/M*255
    img_ini=img_ini/M*255
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
    sin_model=sinmodel.make_model(batch=batch,shape=(736,24,1968))
    ct_model=ctmodel.make_model(shape=(512,512,11))

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    # N=len(img_label[0:-2])
    N=len(img_label)
    vx = sin_ini[::-2]
    vy = [img_label[::-2], sin_label[::-2]]
    
    id=np.array(range(N))
    iter=list(range(0,N,batch))
    tt=np.zeros([N])
    Loss_sin=[]
    Loss_ct=[]
    for i in range(epoch):
        np.random.shuffle(id)
        for j in range(len(iter)):
            u, f, f_noisy = img_label[id[iter[j]:iter[j]+batch]],\
                            sin_label[id[iter[j]:iter[j]+batch]],\
                            sin_ini[id[iter[j]:iter[j]+batch]]
            u,f,f_noisy=tf.constant(u),tf.constant(f),tf.constant(f_noisy)
            tf.debugging.assert_all_finite(u,'u_check')
            tf.debugging.assert_all_finite(f,'f_check')
            tf.debugging.assert_all_finite(f_noisy,'f_noisy_check')
            sin_loss, m2, ct_loss = train_step(f_noisy, sin_model,ct_model, 
                                    [u,f], loss_1, psnr, optimizer,
                                        delt_alpha, delt_theta,
                                        w1, index1, dist, DSD,
                                        h_matrix, brweight, brindex, 
                                        cos_alpha_cor,index5, v_star, 
                                        index_star,t1,t2)
            Loss_sin.append(sin_loss)
            Loss_ct.append(ct_loss)
            np.save('Loss_sin.npy',np.array(Loss_sin))
            np.save('Loss_ct.npy',np.array(Loss_ct))
            tt[iter[j]:iter[j]+batch]=m2[0].numpy()
            print(iter[j], "/", i, ":", "sin_loss",sin_loss.numpy(),
                  "ct_loss:", ct_loss.numpy(),
                  "psnr1", [m2[0].numpy(), m2[1].numpy()],
                #   'psnr3:', [m3[0].numpy(), m3[1].numpy(), m3[2].numpy()]
                  )
        m=0.0
        # ckpt_sin = './ckpt/fp_two_model_unet_sin' 
        # ckpt_ct = './ckpt/fp_two_model_unet_ct'
        ckpt_sin = './ckpt/fp_two_model_unet_sin_2' 
        ckpt_ct = './ckpt/fp_two_model_unet_ct_2'
        if  np.mean(tt)>m:
            m=np.mean(tt)
            sin_model.save_weights(ckpt_sin)
            ct_model.save_weights(ckpt_ct)
    # sin_model.save_weights('./ckpt/two_model_sin')
    # ct_model.save_weights('./ckpt/two_model_ct')


def loss_1(x, y,weights=0.5):
    x0 = tf.cast(x[0], tf.float32)
    x1 = tf.cast(x[1], tf.float32)
    y0 = tf.cast(y[0], tf.float32)
    y1 = tf.cast(y[1], tf.float32)
    return weights*tf.reduce_mean(tf.math.square(x0 - y0))+(1-weights)*tf.reduce_mean(tf.math.square(x1 - y1))
    # return tf.reduce_sum(tf.math.square(x0 - y0)) / shape[0] / shape[1] / shape[2] / shape[3]
    # return tf.reduce_sum(tf.math.square(x1 - y1))/shape1[0] / shape1[1] / shape1[2] / shape1[3]

def psnr(x, y,max_val=255):
    x0 = tf.cast(x[0], tf.float32)
    x1 = tf.cast(x[1], tf.float32)
    y0 = tf.cast(y[0], tf.float32)
    y1 = tf.cast(y[1], tf.float32)
    # batch = tf.cast(tf.shape(x[1])[0], tf.float32)
    psnr1=tf.reduce_mean(tf.image.psnr(x0, y0, max_val=tf.reduce_max(x0)))######psnr of f and de_sin
    # psnr2=tf.reduce_sum(tf.image.psnr(x1, y2, max_val=tf.reduce_max(x1))) / batch######psnr of u and fbp
    psnr3 = tf.reduce_mean(tf.image.psnr(x1, y1, max_val=tf.reduce_max(x1)))#####psnr of u and reconstructed
    return [psnr1,psnr3]

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
    batch=1
    epoch=60
    train(batch,epoch)