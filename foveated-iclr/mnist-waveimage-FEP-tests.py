
# coding: utf-8



# In[2]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf


# In[3]:


#from waveimage import WaveImage, calc_dim, calc_U, mnist_reshape_32
from waveimage import WaveImage, calc_dim, calc_U, mnist_reshape_32


# In[4]:


from scipy.stats import multivariate_normal, entropy


# In[5]:


import math


# In[6]:


import sys, os


# In[7]:


import pickle


# In[8]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


# In[9]:


from waveimage import calc_U


# In[10]:


#DECODER = 'naive-test'
DECODER = 'base' 
if DECODER == 'base':
    mu, Sigma, rho = pickle.load(open("mnist-waveimage-train-mu-Sigma-rho.pkl", "rb"))
elif DECODER == 'naive':
    mu, Sigma, rho = pickle.load(open("mnist-waveimage-train-mu-Sigma-rho-naive.pkl", "rb"))
elif DECODER == 'naive-test':
    mu, Sigma, rho = pickle.load(open("mnist-waveimage-train-mu-Sigma-rho.pkl", "rb"))
#mu, Sigma, rho = pickle.load(open("mnist-waveimage-train-mu-Sigma-rho-noisy-alt.pkl", "rb"))


# In[11]:


#ENCODER = 'base'
ENCODER = 'backbone-CNN-parts' 
if ENCODER == 'backbone-CNN-parts':
    from backbone_CNN_parts_def import *
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    file_name = "models/mnist-waveimage-CNN-backbone-512-rnd-parts/mnist-waveimage-CNN-backbone-512-rnd-parts"
    saver.restore(sess,       file_name + ".ckpt")
    #mem = pickle.load(open(file_name + "_mem.pkl", "rb"))
else:
    sess = None


# ## Creation de la base d'apprentissage

# In[12]:


def wave_tensor_data(batch_x):
    batch_size, _ = batch_x.shape
    wave_tensor = {}
    for h in range(6):
        if h == 0:
            h_size = 1
            wave_tensor[h] = np.zeros((batch_size, h_size, h_size, 1))
        else:
            h_size = 2**(h - 1)
            wave_tensor[h] = np.zeros((batch_size, h_size, h_size, 3))
    for num_batch in range(batch_size):
        image = mnist_reshape_32(batch_x[num_batch])
        w = WaveImage(image = image)
        for h in range(w.get_h_max()):
            data_h = w.get_data()[h]
            if h == 0:
                wave_tensor[h][num_batch][0][0][0] = data_h[(0,0)]
            else:
                for u in data_h:
                    wave_tensor[h][num_batch][u[0]][u[1]][:] = data_h[u]
    return wave_tensor


# In[13]:


def calc_pow2(i_ref):
    pow2_i = np.zeros(5, dtype='int')
    reste = i_ref
    for p in range(4,-1,-1):
        pow2_i[p] = int(reste // 2**p)
        #reste = reste % 2**p
    return pow2_i[::-1]


# In[14]:


print calc_pow2(15)


# In[15]:


def init_wave_tensor(batch_size):
    wave_tensor = {}
    for h in range(6):
        if h == 0:
            h_size = 1
            wave_tensor[h] = np.zeros((batch_size, h_size, h_size, 1))
        else:
            h_size = 2**(h - 1)
            wave_tensor[h] = np.zeros((batch_size, h_size, h_size, 3))
    return wave_tensor
    


# In[16]:


def wave_tensor_data_backbone(batch_x, depth = -1, i_ref = -1, j_ref = -1):
    batch_size, _ = batch_x.shape
    FLAG_RAND_I = i_ref == -1
    FLAG_RAND_J = j_ref == -1
    FLAG_DEPTH = depth == -1
    wave_tensor = init_wave_tensor(batch_size)
    for num_batch in range(batch_size):
        image = mnist_reshape_32(batch_x[num_batch])
        w = WaveImage(image = image)
        if FLAG_RAND_I:
            i_ref = np.random.randint(16)
        if FLAG_RAND_J:
            j_ref = np.random.randint(16)  
        if FLAG_DEPTH:
            depth = 1 + np.random.randint(6)
        pow2_i = calc_pow2(i_ref)
        pow2_j = calc_pow2(j_ref)
        for h in range(6 - depth, 6):
            data_h = w.get_data()[h]
            if h == 0:
                wave_tensor[h][num_batch][0][0][0] = data_h[(0,0)] #/ 4**4
            else:
                u = (pow2_i[h - 1], pow2_j[h - 1])
                #for u in data_h:
                #    wave_tensor[h][num_batch][u[0]][u[1]][:] = 0
                wave_tensor[h][num_batch][u[0]][u[1]][:] = data_h[u] #/ 4 ** (5 - h)
    return wave_tensor


# Construction 
# + 5 couches convolutionnelles : 16 x 16 --> 8 x 8 ; 8 x 8 --> 4 x 4 etc
# + 1 couche FC

# In[17]:


DEPTH_WAV = 3
NB_LABEL = 10


# In[18]:


# Likelihood calculation (over z's, for given v, h and u)

def calc_lik(v,h,u):
    lik = np.zeros(10)
    for c in range(10):
        if np.linalg.norm(v) < 1e-16:
            if np.linalg.norm(mu[c][h][u]) > 1e-16:
                lik[c] = rho[c][h][u]
            else:
                lik[c] = 1                
        else:
            if np.linalg.norm(mu[c][h][u]) > 1e-16:
                if h == 0:
                    dist = multivariate_normal(mean = mu[c][h][u], cov = Sigma[c][h][u])
                else:
                    dist = multivariate_normal(mean = mu[c][h][u], cov = Sigma[c][h][u] + 1e-10 * np.eye(3))
                lik[c] = (1-rho[c][h][u]) * dist.pdf(v)
                #lik[c] =  dist.pdf(v)
            else:
                lik[c] = 0
        lik[c] = max(lik[c],1e-16)    
    return lik


# In[19]:


# Likelihood calculation (over z's, for given v, h and u)

def calc_lik_naive(v,h,u):
    lik = np.zeros(10)
    for c in range(10):
        '''if np.linalg.norm(v) < 1e-16:
            if np.linalg.norm(mu[c][h][u]) > 1e-16:
                lik[c] = rho[c][h][u]
            else:
                lik[c] = 1                
        else:
            if np.linalg.norm(mu[c][h][u]) > 1e-16:'''
        if h == 0:
            dist = multivariate_normal(mean = mu[c][h][u], cov = Sigma[c][h][u])
        else:
            dist = multivariate_normal(mean = mu[c][h][u], cov = np.diag(Sigma[c][h][u] + 1e-10))
            #lik[c] = (1-rho[c][h][u]) * dist.pdf(v)
            lik[c] =  dist.pdf(v)
        '''else:
            lik[c] = 0'''
        lik[c] = max(lik[c], 1e-16)    
    return lik


# In[20]:


# Log posterior

def update_log_score(log_score, lik):
    #print 'lik =' + str(lik) 
    log_score += np.log(lik) 
    max_log_score = max(log_score)
    log_score -= max_log_score
    return log_score


# In[21]:


# Posterior (Softmax)
    
def calc_pi(log_score): # TODO
    Z = np.sum(np.exp(log_score))
    pi = np.exp(log_score)/Z
    #print 'pi =' + str(pi)
    #print 'max(pi) = ',max(pi)
    return pi


# # Parcours predictif

# In[22]:


pi_predictive_eff = pickle.load(open("mnist-waveimage-saliency-map.pkl", "rb"))    
#pi_predictive_eff = pickle.load(open("mnist-waveimage-saliency-map-diff-backbone-CNN-parts.pkl", "rb"))    


# In[23]:


H_generic_eff = pickle.load(open("mnist-waveimage-generic-saliency-map.pkl", "rb"))    
#pi_predictive_eff = pickle.load(open("mnist-waveimage-saliency-map-diff-backbone-CNN-parts.pkl", "rb"))    


# In[24]:


def calcule_asc_path(h,u):
    rep = []
    for h_inf in range(h, 0, -1):
        i_inf = u[0] / (2 ** (h - h_inf))
        j_inf = u[1] / (2 ** (h - h_inf))
        rep += [(h_inf, (i_inf, j_inf))]
    # racine
    rep += [(0, (i_inf, j_inf))]
    return rep


# In[25]:


def calc_pi_predictive_sorted(pi_predictive): 
    pi_predictive_sorted = {}
    for c in range(10):
        pi_predictive_sorted[c] = []
        for k in pi_predictive[c]:
            pi_predictive_sorted[c] += [(pi_predictive[c][k], k)]
        pi_predictive_sorted[c] = sorted(pi_predictive_sorted[c])
    return pi_predictive_sorted


# In[26]:


def calc_H_predictive_sorted(H_predictive): 
    H_predictive_sorted = []
    for k in H_predictive:
        H_predictive_sorted += [(H_predictive[k], k)]
    H_predictive_sorted = sorted(H_predictive_sorted, reverse=True)
    return H_predictive_sorted


# In[27]:


def argmax_generator(c, h, u):
    test_pred = rho[c][h][u] < .5       
    if test_pred:
        return mu[c][h][u]
    else:
        return np.zeros(3)


# In[28]:


def softmax_generator(log_score, h, u):
    Z = np.sum(np.exp(log_score))
    mu_c = np.zeros(3)
    for c in range(NB_LABEL):
        pi = np.exp(log_score[0][c]) / Z
        mu_c += pi * mu[c][h][u] * (1 - rho[c][h][u])
    return mu_c


# In[29]:


axes = []
h_max = 6
shape = (32,32)

U = {}
for h in range(h_max):
    U [h] = {}
    dim_i, dim_j = calc_dim(shape, h, h_max)
    for i in range(dim_i):
        for j in range(dim_j):
            U[h][(i,j)] = 1    


# In[30]:


DECODER


# h_max = 6
# lik_predictive = {}
# for c in range(10):
#     lik_predictive[c] = {}
#     for h in range(h_max):
#         lik_predictive[c][h] = {}
#         for u in U[h]:
#             v_predictive = argmax_generator(c, h, u)
#             if MODEL == 'base':
#                 lik = calc_lik(v_predictive, h, u)
#             elif MODEL == 'naive':
#                 lik = calc_lik_naive(v_predictive, h, u)
#             lik_predictive[c][h][u] = lik

# In[31]:


h_max = 6
if DECODER == 'base':
    lik_predictive_base = {}
elif DECODER == 'naive' or DECODER == 'naive-test':  
    lik_predictive_naive = {}
for c in range(10):
    if DECODER == 'base':
        lik_predictive_base[c] = {}
    elif DECODER == 'naive' or DECODER == 'naive-test':  
        lik_predictive_naive[c] = {}
    for h in range(h_max):
        if DECODER == 'base':
            lik_predictive_base[c][h] = {}
        elif DECODER == 'naive' or DECODER == 'naive-test':  
            lik_predictive_naive[c][h] = {}
        for u in U[h]:
            v_predictive = argmax_generator(c, h, u)
            if DECODER == 'base':
                lik = calc_lik(v_predictive, h, u)
                lik_predictive_base[c][h][u] = lik
            elif DECODER == 'naive' or DECODER == 'naive-test':
                lik = calc_lik_naive(v_predictive, h, u)
                lik_predictive_naive[c][h][u] = lik


# In[32]:


def predictive_search(z_ref, log_score, actions_set, mem_h_u, FLAG_POL = 'sharp-predictive-Info-Gain'):
    # actions_set ne contient que les positions de niveau 5
    h_ref = 5
    batch_size = len(actions_set)
                
    #print batch_size
    ## Parcours predictif
    log_score_path = np.zeros((batch_size, NB_LABEL))
    for i, u_gen in enumerate(actions_set):
        log_score_path[i] = np.copy(log_score[0])
        
    dict_u = {}
    for i, u_gen in enumerate(actions_set):
        dict_u[i] = u_gen
        liste_path = calcule_asc_path(h_ref, u_gen)
        #print(liste_path)
        for (h_path, u_path) in liste_path[:-1]:
            #print(h_path, u_path)
            if (h_path, u_path) not in mem_h_u:
                log_score_path[i] = update_log_score(log_score_path[i], lik_predictive[z_ref][h_path][u_path])
        #print log_score_path[i]
        
    q_pre = np.exp(log_score[0]) / np.sum(np.exp(log_score[0]))
    FEP_post = np.zeros(batch_size)
    for i, u_gen in enumerate(actions_set):
        if FLAG_POL != 'sharp-predictive-Info-Gain':
            q_post = np.exp(log_score_path[i]) / np.sum(np.exp(log_score_path[i])) 
            if FLAG_POL == 'sharp-predictive-Infomax':
                FEP_post[i] = entropy(q_post)
            elif FLAG_POL == 'sharp-predictive-Innovation':
                FEP_post[i] = - entropy(q_post, q_pre)
            elif FLAG_POL == 'sharp-predictive-Conservation':
                FEP_post[i] = - np.log(q_post[z_ref])
            elif FLAG_POL == 'sharp-predictive-IG-post':
                delta_log_score_path = log_score_path[i] - log_score[0]
                delta_q_post = np.exp(delta_log_score_path) / np.sum(np.exp(delta_log_score_path))
                FEP_post[i] = entropy(q_post) + entropy(q_post, delta_q_post )    
        else: # FLAG_POL == 'sharp-predictive-Info-Gain'
            delta_log_score_path = log_score_path[i] - log_score[0]
            delta_q_post = np.exp(delta_log_score_path) / np.sum(np.exp(delta_log_score_path))
            FEP_post[i] = - np.log(delta_q_post[z_ref])
        if not np.isfinite(FEP_post[i]):
            print "aie!!"
            FEP_post[i] = 50
                
    #i_max = np.where(log_score_path[:, z_ref] == max(log_score_path[:, z_ref]))[0][0]
    i_min = np.where(FEP_post == min(FEP_post))[0][0]
    q_post = np.exp(log_score_path[i_min]) / np.sum(np.exp(log_score_path[i_min])) 
    delta_log_score_path = log_score_path[i_min] - log_score[0]
    delta_q_post = np.exp(delta_log_score_path) / np.sum(np.exp(delta_log_score_path))
    #print log_score_path[i_min] 
    #print dict_u[i_min] , entropy(q_post), - np.log(delta_q_post[z_ref]) + np.log(q_pre[z_ref]), FEP_post[i_min]
    
    
    ## 3 ##
    return dict_u[i_min] #, pi_path[i_max] #pi_path[i_max][z_ref]
    


# In[33]:


def predictive_search_CNN(sess, wave_tensor, z_ref, log_score, actions_set, mem_h_u, FLAG_POL = 'sharp-predictive-Info-Gain'):
    # actions_set ne contient que les positions de niveau 5
    h_ref = 5
    batch_size = len(actions_set)
    
    batch_predictive_tensor =  {} #init_wave_tensor(batch_size)
    for h in range(6):
        if h == 0:
            h_size = 1
            batch_predictive_tensor[h] = np.zeros((batch_size, h_size, h_size, 1))
        else:
            h_size = 2**(h - 1)
            batch_predictive_tensor[h] = np.zeros((batch_size, h_size, h_size, 3))
        for i in range(batch_size):
            batch_predictive_tensor[h][i] = np.copy(wave_tensor[h][0])
                
    #print batch_size
    ## Parcours predictif
    dict_u = {}
    for i, u_gen in enumerate(actions_set):
        dict_u[i] = u_gen
        liste_path = calcule_asc_path(h_ref, u_gen)
        #print(liste_path)
        for (h_path, u_path) in liste_path[:-1]:
            #print(h_path, u_path)
            if (h_path, u_path) not in mem_h_u:
                v_predictive = argmax_generator(z_ref, h_path, u_path)
                batch_predictive_tensor[h_path][i][u_path[0]][u_path[1]][:] =  v_predictive
                
        #print log_score_path[i]
    log_score_path = y_hat_logit.eval(feed_dict={ x_5: batch_predictive_tensor[5],                                        x_4: batch_predictive_tensor[4],                                        x_3: batch_predictive_tensor[3],                                        x_2: batch_predictive_tensor[2],                                        x_1: batch_predictive_tensor[1],                                        x_0: batch_predictive_tensor[0],                                        keep_prob: 1,                                        batch_phase:False})    
        
    q_pre = np.exp(log_score[0]) / np.sum(np.exp(log_score[0]))
    FEP_post = np.zeros(batch_size)
    for i, u_gen in enumerate(actions_set):
        if FLAG_POL != 'sharp-predictive-Info-Gain':
            q_post = np.exp(log_score_path[i]) / np.sum(np.exp(log_score_path[i])) 
            if FLAG_POL == 'sharp-predictive-Infomax':
                FEP_post[i] = entropy(q_post)
            elif FLAG_POL == 'sharp-predictive-Innovation':
                FEP_post[i] = - entropy(q_post, q_pre)
            elif FLAG_POL == 'sharp-predictive-Conservation':
                FEP_post[i] = - np.log(q_post[z_ref])
            elif FLAG_POL == 'sharp-predictive-IG-post':
                delta_log_score_path = log_score_path[i] - log_score[0]
                delta_q_post = np.exp(delta_log_score_path) / np.sum(np.exp(delta_log_score_path))
                FEP_post[i] = entropy(q_post) + entropy(q_post, delta_q_post )    
        else: # FLAG_POL == 'sharp-predictive-Info-Gain'
            delta_log_score_path = log_score_path[i] - log_score[0]
            delta_q_post = np.exp(delta_log_score_path) / np.sum(np.exp(delta_log_score_path))
            FEP_post[i] = - np.log(delta_q_post[z_ref])
        if not np.isfinite(FEP_post[i]):
            print "aie!!"
            FEP_post[i] = 50
                
    #i_max = np.where(log_score_path[:, z_ref] == max(log_score_path[:, z_ref]))[0][0]
    i_min = np.where(FEP_post == min(FEP_post))[0][0]
    q_post = np.exp(log_score_path[i_min]) / np.sum(np.exp(log_score_path[i_min])) 
    delta_log_score_path = log_score_path[i_min] - log_score[0]
    delta_q_post = np.exp(delta_log_score_path) / np.sum(np.exp(delta_log_score_path))
    #print log_score_path[i_min] 
    #print dict_u[i_min] , entropy(q_post), - np.log(delta_q_post[z_ref]) + np.log(q_pre[z_ref]), FEP_post[i_min]
    
    
    ## 3 ##
    return dict_u[i_min] #, pi_path[i_max] #pi_path[i_max][z_ref]
    


# In[34]:


def FEP_predictive_search(log_score, actions_set, mem_h_u, FLAG_POL = 'smooth-predictive-Info-Gain'):
    # actions_set ne contient que les positions de niveau 5
    h_ref = 5
    batch_ref = len(actions_set)
    batch_size = batch_ref * NB_LABEL
    log_score_path = {}
    
    ## Parcours predictif
    log_score_path = np.zeros((batch_size, NB_LABEL))
    for i, u_gen in enumerate(actions_set):
        for c in range(NB_LABEL):
            i_full = c * batch_ref + i
            log_score_path[i_full] = np.copy(log_score[0])
            
    dict_u = {}
    for i, u_gen in enumerate(actions_set):
        dict_u[i] = u_gen
        liste_path = calcule_asc_path(h_ref, u_gen)
        #print(liste_path)
        for (h_path, u_path) in liste_path[:-1]:
            #print(h_path, u_path)
            if (h_path, u_path) not in mem_h_u:
                for c in range(NB_LABEL):
                    i_full = c * batch_ref + i
                    log_score_path[i_full] = update_log_score(log_score_path[i_full],                                                              lik_predictive[c][h_path][u_path])
                        
    FEP_post = np.zeros(batch_size)
    q_pre = np.exp(log_score[0]) / np.sum(np.exp(log_score[0]))
    '''log_score_post_full = np.zeros((batch_ref,))
    for c in range(NB_LABEL):
        log_score_post_full += q_pre[c] * log_score_path[i_full]    #print q_pre'''
        
    for i, u_gen in enumerate(actions_set):
        for c in range(NB_LABEL):
            i_full = c * batch_ref + i
            if FLAG_POL != 'smooth-predictive-Info-Gain':
                q_post = np.exp(log_score_path[i_full]) / np.sum(np.exp(log_score_path[i_full]))  
                if FLAG_POL == 'smooth-predictive-Infomax':
                    FEP_post[i_full] = entropy(q_post)
                elif FLAG_POL == 'smooth-predictive-Innovation':
                    FEP_post[i_full] = - entropy(q_post, q_pre)
                elif FLAG_POL == 'smooth-predictive-Conservation':
                    FEP_post[i_full] = - np.log(q_post[c])
                elif FLAG_POL == 'smooth-predictive-IG-post':
                    delta_log_score_path = log_score_path[i_full] - log_score[0]
                    delta_q_post = np.exp(delta_log_score_path) / np.sum(np.exp(delta_log_score_path))
                    FEP_post[i_full] = entropy(q_post) + entropy(q_post, delta_q_post )    
            else: # FLAG_POL == 'smooth-predictive-Info-Gain'
                delta_log_score_path = log_score_path[i_full] - log_score[0]
                delta_q_post = np.exp(delta_log_score_path) / np.sum(np.exp(delta_log_score_path))
                FEP_post[i_full] = - np.log(delta_q_post[c])
            if not np.isfinite(FEP_post[i_full]):
                print "aie!!"
                FEP_post[i_full] = 50
                
    #ch = raw_input('')    
    FEP_post_full = np.zeros((batch_ref,))
    for c in range(NB_LABEL):
        FEP_post_full += q_pre[c] * FEP_post[c * batch_ref : (c + 1) * batch_ref]
        #print FEP_post[c * batch_ref : (c + 1) * batch_ref]
        
    #for i, u_gen in enumerate(actions_set):
    #    print u_gen, FEP_post_full[i]
    
    #print FEP_post_full
    i_min = np.where(FEP_post_full == min(FEP_post_full))[0][0]
    #print i_min, dict_u[i_min], FEP_post_full[i_min]
    ## 3 ##
    return dict_u[i_min]  #pi_path[i_max][z_ref]
    


# In[35]:


def FEP_predictive_search_CNN(sess, wave_tensor, log_score, actions_set, mem_h_u, FLAG_POL = 'smooth-predictive-Info-Gain'):
    # actions_set ne contient que les positions de niveau 5
    h_ref = 5
    batch_ref = len(actions_set)
    batch_size = batch_ref * NB_LABEL
    log_score_path = {}
    
    batch_predictive_tensor =  {} #init_wave_tensor(batch_size)
    for h in range(6):
        if h == 0:
            h_size = 1
            batch_predictive_tensor[h] = np.zeros((batch_size, h_size, h_size, 1))
        else:
            h_size = 2**(h - 1)
            batch_predictive_tensor[h] = np.zeros((batch_size, h_size, h_size, 3))
        for i in range(batch_size):
            batch_predictive_tensor[h][i] = np.copy(wave_tensor[h][0])
    
    ## Parcours predictif
    dict_u = {}
    for i, u_gen in enumerate(actions_set):
        dict_u[i] = u_gen
        liste_path = calcule_asc_path(h_ref, u_gen)
        #print(liste_path)
        for (h_path, u_path) in liste_path[:-1]:
            #print(h_path, u_path)
            if (h_path, u_path) not in mem_h_u:
                for c in range(NB_LABEL):
                    i_full = c * batch_ref + i
                    v_predictive = argmax_generator(c, h_path, u_path)
                    batch_predictive_tensor[h_path][i_full][u_path[0]][u_path[1]][:] =  v_predictive
     
    log_score_path = y_hat_logit.eval(feed_dict={ x_5: batch_predictive_tensor[5],                                    x_4: batch_predictive_tensor[4],                                    x_3: batch_predictive_tensor[3],                                    x_2: batch_predictive_tensor[2],                                    x_1: batch_predictive_tensor[1],                                    x_0: batch_predictive_tensor[0],                                    keep_prob: 1,                                    batch_phase:False})    
    
    
    FEP_post = np.zeros(batch_size)
    q_pre = np.exp(log_score[0]) / np.sum(np.exp(log_score[0]))
        
    for i, u_gen in enumerate(actions_set):
        for c in range(NB_LABEL):
            i_full = c * batch_ref + i
            if FLAG_POL != 'smooth-predictive-Info-Gain':
                q_post = np.exp(log_score_path[i_full]) / np.sum(np.exp(log_score_path[i_full]))  
                if FLAG_POL == 'smooth-predictive-Infomax':
                    FEP_post[i_full] = entropy(q_post)
                elif FLAG_POL == 'smooth-predictive-Innovation':
                    FEP_post[i_full] = - entropy(q_post, q_pre)
                elif FLAG_POL == 'smooth-predictive-Conservation':
                    FEP_post[i_full] = - np.log(q_post[c])
                elif FLAG_POL == 'smooth-predictive-IG-post':
                    delta_log_score_path = log_score_path[i_full] - log_score[0]
                    delta_q_post = np.exp(delta_log_score_path) / np.sum(np.exp(delta_log_score_path))
                    FEP_post[i_full] = entropy(q_post) + entropy(q_post, delta_q_post )    
            else: # FLAG_POL == 'smooth-predictive-Info-Gain'
                delta_log_score_path = log_score_path[i_full] - log_score[0]
                delta_q_post = np.exp(delta_log_score_path) / np.sum(np.exp(delta_log_score_path))
                FEP_post[i_full] = - np.log(delta_q_post[c])
            if not np.isfinite(FEP_post[i_full]):
                print "aie!!"
                FEP_post[i_full] = 50
                
    #ch = raw_input('')    
    FEP_post_full = np.zeros((batch_ref,))
    for c in range(NB_LABEL):
        FEP_post_full += q_pre[c] * FEP_post[c * batch_ref : (c + 1) * batch_ref]
        #print FEP_post[c * batch_ref : (c + 1) * batch_ref]
        
    #for i, u_gen in enumerate(actions_set):
    #    print u_gen, FEP_post_full[i]
    
    #print FEP_post_full
    i_min = np.where(FEP_post_full == min(FEP_post_full))[0][0]
    #print i_min, dict_u[i_min], FEP_post_full[i_min]
    ## 3 ##
    return dict_u[i_min]  #pi_path[i_max][z_ref]
    


# In[36]:


def prediction_based_policy(log_score, actions_set, mem_h_u, FLAG_POL = 'sharp-predictive-Info-Gain'):
    
    ## 1 ##
    z_tilde = np.argmax(log_score)    
    u_tilde = predictive_search(z_tilde, log_score, actions_set, mem_h_u, FLAG_POL)
    
    return u_tilde


# In[49]:


def prediction_based_policy_CNN(sess, wave_tensor, log_score, actions_set, mem_h_u, FLAG_POL = 'sharp-predictive-Info-Gain'):
    
    ## 1 ##
    z_tilde = np.argmax(log_score)    
    u_tilde = predictive_search_CNN(sess, wave_tensor, z_tilde, log_score, actions_set, mem_h_u, FLAG_POL)
    
    return u_tilde


# In[38]:


def FEP_prediction_based_policy(log_score, actions_set, mem_h_u, FLAG_POL = 'smooth-predictive-Info-Gain'):
    
    u_tilde = FEP_predictive_search(log_score, actions_set, mem_h_u, FLAG_POL)
    
    return u_tilde


# In[50]:


def FEP_prediction_based_policy_CNN(sess, wave_tensor, log_score, actions_set, mem_h_u, FLAG_POL = 'smooth-predictive-Info-Gain'):
    
    u_tilde = FEP_predictive_search_CNN(sess, wave_tensor, log_score, actions_set, mem_h_u, FLAG_POL)
    
    return u_tilde


# In[40]:


def saliency_based_policy(log_score, pi_predictive_sorted, mem_h_u):
    h = 5
    ## 1 ##
    z_tilde = np.argmax(log_score)
    u_tilde = pi_predictive_sorted[z_tilde].pop()[1]
    while (h, u_tilde) in mem_h_u:
        u_tilde = pi_predictive_sorted[z_tilde].pop()[1]
    return u_tilde
    


# In[41]:


def generic_saliency_based_policy(H_predictive_sorted, mem_h_u):
    u_tilde = H_predictive_sorted.pop()[1]
    return u_tilde


# In[42]:


def random_policy(log_score, mem_h_u):
    h = 5
    ## 1 ##
    z_tilde = np.argmax(log_score)
    #u_tilde = (1 + np.random.randint(14),  1 + np.random.randint(14))
    u_tilde = (np.random.randint(16),  np.random.randint(16))
    while (h, u_tilde) in mem_h_u:
        #u_tilde = (1 + np.random.randint(14),  1 + np.random.randint(14))
        u_tilde = (np.random.randint(16),  np.random.randint(16))
    return u_tilde


# In[43]:


def scene_exploration(sess, wave_tensor_ref, wave_tensor, log_score, z_ref, ind_test, actions_set, mem_h_u, record,                       POL = 'predictive', AFF = True, THRESHOLD = 1e-4, INIT = 'limit'):
    
    assert POL == 'sharp-predictive-Info-Gain' or POL == 'sharp-predictive-Infomax'    or POL == 'sharp-predictive-Innovation' or POL == 'sharp-predictive-Conservation' or POL == 'sharp-predictive-IG-post'    or POL == 'saliency-based' or POL == 'random' or POL == 'full' or POL == 'generic-saliency-based'    or POL == 'smooth-predictive-Info-Gain' or POL == 'smooth-predictive-Infomax'    or POL == 'smooth-predictive-Innovation' or POL == 'smooth-predictive-Conservation' or POL == 'smooth-predictive-IG-post'
    
    if POL == 'full':
        THRESHOLD = 0
        POL = 'generic-saliency-based'
    
    TOUR = 0
    END = False
    h_ref = 5
    
    # saliency-based approach
    if POL == 'saliency-based':
        pi_predictive_sorted = calc_pi_predictive_sorted(pi_predictive_eff)
    if True : #POL == 'generic-saliency-based':
        H_predictive_sorted = calc_H_predictive_sorted(H_generic_eff)     
    while END == False:
        
        if AFF:
            print '************************************'
            print '******       TOUR    ' + str(TOUR + 1) + '        ******'
            print '************************************'
        
        # 1. CHOIX
        if TOUR == 0 and INIT == 'H0-init':
            u_tilde = H_predictive_sorted[-1][1]
            H_predictive_sorted.pop()
        else:
            if POL == 'sharp-predictive-Info-Gain' or POL == 'sharp-predictive-Infomax'            or POL == 'sharp-predictive-Innovation' or POL == 'sharp-predictive-Conservation' or POL == 'sharp-predictive-IG-post':
                if ENCODER == 'backbone-CNN-parts':
                    u_tilde = prediction_based_policy_CNN(sess, wave_tensor, log_score, actions_set, mem_h_u, FLAG_POL = POL)
                else:
                    u_tilde = prediction_based_policy(log_score, actions_set, mem_h_u, FLAG_POL = POL)
            elif POL == 'smooth-predictive-Info-Gain' or POL == 'smooth-predictive-Infomax'              or POL == 'smooth-predictive-Innovation' or POL == 'smooth-predictive-Conservation' or POL == 'smooth-predictive-IG-post':
                if ENCODER == 'backbone-CNN-parts':
                    u_tilde = FEP_prediction_based_policy_CNN(sess, wave_tensor, log_score, actions_set, mem_h_u, FLAG_POL = POL)
                else:
                    u_tilde = FEP_prediction_based_policy(log_score, actions_set, mem_h_u, FLAG_POL = POL)    
            elif POL == 'saliency-based' :
                u_tilde = saliency_based_policy(log_score, pi_predictive_sorted, mem_h_u)
            elif POL == 'generic-saliency-based':
                u_tilde = generic_saliency_based_policy(H_predictive_sorted, mem_h_u)
            else:
                u_tilde = random_policy(log_score, mem_h_u)
        if AFF:
            print 'CHOIX :', u_tilde
        
        # 2. LECTURE + UPDATE
        #wave_tensor =  init_wave_tensor(1)
        liste_path = calcule_asc_path(h_ref, u_tilde)
        
        for (h_path, u_path) in reversed(liste_path):
            if (h_path, u_path) not in mem_h_u:
                if ENCODER == 'backbone-CNN-parts':
                    wave_tensor[h_path][0][u_path[0]][u_path[1]][:] =  wave_tensor_ref[h_path][0][u_path[0]][u_path[1]][:]
                else:
                    v = wave_tensor_ref[h_path][0][u_path[0]][u_path[1]][:]
                    lik = calc_lik(v, h_path, u_path)
                    log_score[0] = update_log_score(log_score[0], lik)
                #wave_tensor[h_path][0][u_path[0]][u_path[1]][:] =  
                mem_h_u += [(h_path, u_path)]
                record.mem_h_u += [(h_path, u_path)]
                record.nb_coeffs += 3  
        
        if ENCODER == 'backbone-CNN-parts':
            log_score = y_hat_logit.eval(feed_dict={x_5: wave_tensor[5],                            x_4: wave_tensor[4],                            x_3: wave_tensor[3],                            x_2: wave_tensor[2],                            x_1: wave_tensor[1],                            x_0: wave_tensor[0],                            keep_prob: 1,                            batch_phase:False}) 
        
        pi = np.exp(log_score[0]) / np.sum(np.exp(log_score[0])) #sess.run(tf.nn.softmax(log_score))[0]
                
        H = entropy(pi) # sess.run(tf.nn.softmax_cross_entropy_with_logits(labels=pi,logits=log_score)) #np.sum(- pi * np.log(pi))
        out = np.argmax(pi)
                
        if AFF :
            print 'pi : ', pi
            print 'out :', out
            print 'pi[out] : ', pi[out]
            print 'H : ', H

        record.mem_pi += [pi]
        record.mem_H += [H]
        record.mem_z += [out]
                
        # 3. INHIBITION OF RETURN        
        actions_set.pop(u_tilde)
        
        record.mem_u += [u_tilde]
        record.nb_saccades += 1
                
        if AFF:
            print '****', 'z :', z_ref, ', u :',u_tilde, ' ---> ', out
              
        
        if TOUR == 16 * 16 - 1 or H < THRESHOLD:
            END = True
            if AFF :
                print '************************************'
                print '******         FINI          *******'
                print '************************************' 
            return out
        else:
            TOUR += 1   


# ## Main

# In[44]:


from record import Record, affiche_records            


# In[45]:


NB_TRIALS = 10000


# In[46]:


mnist = input_data.read_data_sets("MNIST_data/")


#     Policy : predictive, threshold : 0.03, 4 saccades, initial : 3, final : 7, classe : 7, elapsed time : 4.96038
#     Policy : predictive, threshold : 0.03, 4 saccades, initial : 0, final : 2, classe : 2, elapsed time : 9.72888
#     Policy : predictive, threshold : 0.03, 1 saccades, initial : 1, final : 1, classe : 1, elapsed time : 11.6165
#     Policy : predictive, threshold : 0.03, 2 saccades, initial : 0, final : 0, classe : 0, elapsed time : 14.5421
#     Policy : predictive, threshold : 0.03, 8 saccades, initial : 4, final : 4, classe : 4, elapsed time : 22.9571
#     Policy : predictive, threshold : 0.03, 1 saccades, initial : 1, final : 1, classe : 1, elapsed time : 24.8637
#     Policy : predictive, threshold : 0.03, 3 saccades, initial : 7, final : 4, classe : 4, elapsed time : 28.7859
#     Policy : predictive, threshold : 0.03, 4 saccades, initial : 8, final : 9, classe : 9, elapsed time : 33.8153
#     Policy : predictive, threshold : 0.03, 10 saccades, initial : 2, final : 5, classe : 5, elapsed time : 44.5951
#     Policy : predictive, threshold : 0.03, 3 saccades, initial : 0, final : 9, classe : 9, elapsed time : 48.4158

# In[47]:


# Test generic saliency map


# In[52]:


import time
dict_records = {}

#file_name = "mnist-waveimage-CNN-backbone-records-rnd-parts-generic-saliency.npy"
file_name = "mnist-waveimage-records-H0_init-" + ENCODER + '-' + DECODER + ".npy" #random.npy" #-naive-bayes.npy"
#file_name = "mnist-waveimage-records-FEP-dual-full-naive-bayes.npy"
#file_name = "tmp"

INIT = 'H0-init' #

if DECODER == 'base':
    lik_predictive = lik_predictive_base
elif DECODER == 'naive' or DECODER == 'naive-test':
    lik_predictive = lik_predictive_naive

liste_pol  = (            'smooth-predictive-Info-Gain', 'smooth-predictive-Infomax',             'smooth-predictive-Innovation', 'smooth-predictive-Conservation', 'smooth-predictive-IG-post',             'saliency-based', 'generic-saliency-based', 'random',             #'full', \
            'sharp-predictive-Info-Gain', 'sharp-predictive-Infomax', \
            'sharp-predictive-Innovation', 'sharp-predictive-Conservation', 'sharp-predictive-IG-post',\
             )
    
if not os.path.isfile(file_name):
    for POL in liste_pol:
        dict_records[POL] = {}

        #for THRESHOLD in (2, 1.5, 1, 7e-1, 5e-1, 3e-1, 2e-1, 1e-1):# 
        if ENCODER == 'backbone-CNN-parts':
            liste_threshold = (1, 3e-1, 1e-1, 3e-2, 1e-2)
        else:
            liste_threshold = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
            
        for THRESHOLD in liste_threshold: 

            records = [] 
            cpt_TRIALS = 0

            tic = time.time()

            for ind_test in range(NB_TRIALS):
                if ind_test % 10 == 0:
                    print(POL, THRESHOLD, ind_test)
                x_test, z_ref = mnist.test.images[ind_test], mnist.test.labels[ind_test]
                wave_tensor_ref = wave_tensor_data(np.reshape(x_test , (1, 28*28)))
                          # 
                # initial
                log_score = np.zeros((1,10))
                pi = np.ones(10) / 10
                H = entropy(pi) #np.sum(- pi * np.log(pi))

                record = Record()
                record.POL = POL
                record.THRESHOLD = THRESHOLD
                record.z_ref = z_ref
                record.mem_pi += [pi]
                record.mem_H += [H]
                z_tilde = -1

                mem_h_u = []
                wave_tensor =  init_wave_tensor(1)

                # initial actions set
                actions_set = {}
                for i in range(16):
                    for j in range(16):
                        actions_set[(i, j)] = 1
                
                if ind_test == 0:
                    AFF = True
                else:
                    AFF = False

                z_final = scene_exploration(sess, wave_tensor_ref, wave_tensor, log_score, z_ref, ind_test,                                            actions_set, mem_h_u, record,                                             POL = POL, AFF = AFF, INIT = INIT, THRESHOLD = THRESHOLD)
                record.z_final = z_final
                record.success = z_ref == z_final

                records += [record]

                toc = time.time()
                if NB_TRIALS <= 1000:
                    print '\rPolicy : %s, threshold : %g, %d saccades, initial : %d, final : %d, classe : %d, elapsed time : %g'                                     % (POL, THRESHOLD, record.nb_saccades, z_tilde, z_final, z_ref, toc - tic)   
                cpt_TRIALS  += NB_TRIALS

            dict_records[POL][THRESHOLD] = records
            print '\n'
            print 'Nb trials :', cpt_TRIALS
            #affiche_records(records)
            print '\n'
            np.save(file_name, dict_records)
else:
    dict_records = np.load(file_name)
        

