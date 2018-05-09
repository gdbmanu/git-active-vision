
# coding: utf-8


# In[3]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# In[4]:


#from waveimage import WaveImage, calc_dim, calc_U, mnist_reshape_32
from waveimage import WaveImage, calc_dim, calc_U, mnist_reshape_32


# In[5]:


from scipy.stats import multivariate_normal


# In[6]:


import math


# In[7]:


import sys, os


# In[8]:


import pickle


# In[9]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")


# In[10]:


from waveimage import calc_U


# ## Creation de la base d'apprentissage

# In[12]:


file_name = "mnist-waveimage-train-mu-Sigma-rho.pkl"
if not os.path.isfile(file_name):
    
    B_train = []
    for i in range(len(mnist.train.images)):
        if i % 1000 == 0 :
            sys.stdout.write('\rstep %d' % i) 
            sys.stdout.flush()
        c = mnist.train.labels[i]
        x_ref = mnist.train.images[i]
        image = mnist_reshape_32(x_ref)
        w = WaveImage(image = image)
        data = w.get_data()
        for h in range(w.get_h_max()):
            data_h = w.get_data()[h]
            for u in data_h:
                v = data_h[u]
                B_train += [(v,(c,h,u))]   
                
    ### Dictionnaire (Base d'apprentissage)    
    
    Data_train = [[],[],[],[],[],[],[],[],[],[]]
    for c in range(10):
        Data_train[c] = [{},{},{},{},{},{}] 

    for d in B_train:
        v = d[0]
        c = d[1][0]
        h = d[1][1]
        u = d[1][2]
        if u in Data_train[c][h]:
            Data_train[c][h][u] += [v]
        else:
            Data_train[c][h][u] = [v]
            
    ### Probas elementaires         
    mu = []
    Sigma = []
    rho = []
    for c in range(10):
        mu += [{}]
        Sigma += [{}]
        rho += [{}]
        for h in range(0,6):
            mu[c][h] = {}
            Sigma[c][h] = {}
            rho[c][h] = {}
            for u in calc_U((32, 32), h, 6):
                if u in Data_train[c][h]:
                    data = []
                    cpt = 0
                    for v in Data_train[c][h][u]:
                        if np.linalg.norm(v) < 1e-16:
                            cpt += 1
                        else:
                            data += [v]
                    #if h == -1 :
                    #    print len(data)
                    if len(data) > 1:
                        mu[c][h][u] = np.mean(data, 0) #Data[c][h][u],0)
                        Sigma[c][h][u] = np.cov(np.array(data).T) #Data[c][h][u]).T)
                        rho[c][h][u] = float(cpt) / len(Data_train[c][h][u])
                    else:
                        mu[c][h][u] = np.zeros(3)
                        Sigma[c][h][u] = np.zeros((3,3))
                        rho[c][h][u] = 1.
    del B_train, Data_train 
    pickle.dump((mu, Sigma, rho),  open(file_name, "wb"))
else:
    mu, Sigma, rho = pickle.load(open(file_name, "rb"))


# #### Liste des coordonnées par niveau : U[h], h $\in$ 0..5

# In[11]:


axes = []
h_max = 6
shape = (32,32)

U = {}
for h in range(h_max):
    #U_ref = {}
    #for pos_i in axes[h]:
    #    for pos_j in axes[h]:
    #        U_ref[h] += [(pos_i, pos_j)]
    U [h] = {}
    dim_i, dim_j = calc_dim(shape, h, h_max)
    for i in range(dim_i):
        for j in range(dim_j):
            U[h][(i,j)] = 1    
    print 'Niveau ', h, ' : '
    print ' U[' , h, '] :', U[h]
    #print ' U_ref[h] :', U_ref[h]
    print ''
    


# #### Construction d'un arbre de coordonnées multi-niveau (descendants pour (i,j) au niveau h)

# In[12]:


def fils_rec(shape, h, h_max, i, j):
    
    if h < h_max :
        dim_i, dim_j = calc_dim(shape, h, h_max)
        if i < dim_i and j < dim_j :
            rep = [(h,(i,j)), [], [], [], []]
            rep[1] = fils_rec(shape, h + 1, h_max, i * 2, j * 2)
            rep[2] = fils_rec(shape, h + 1, h_max, i * 2, j * 2 + 1)
            rep[3] = fils_rec(shape, h + 1, h_max, i * 2 + 1, j * 2)
            rep[4] = fils_rec(shape, h + 1, h_max, i * 2 + 1, j * 2 + 1)
        else:
            rep = []
    else:
        rep = []
    return rep
    


# In[13]:


U_tree = fils_rec(shape, 0, h_max, 0, 0) 
print U_tree


# #### Calcul des descendants et des parents

# In[14]:


def calcule_desc(U_tree, mem_h_u_todo):
    if U_tree == []:
        return []
    else :
        if U_tree[0] in mem_h_u_todo :
            rep = [U_tree[0]]
        else:
            rep = []
        if U_tree[1] != [] :
            rep += calcule_desc(U_tree[1], mem_h_u_todo)
        if U_tree[2] != [] :
            rep += calcule_desc(U_tree[2], mem_h_u_todo)
        if U_tree[3] != [] :
            rep += calcule_desc(U_tree[3], mem_h_u_todo)
        if U_tree[4] != [] :
            rep += calcule_desc(U_tree[4], mem_h_u_todo)    
        return rep


# In[15]:


print calcule_desc(U_tree, [(5, (5, 7)), (4, (2, 3))])


# In[16]:


def find_desc (U_tree, (h, u)):
    if U_tree == []:
        return None
    else :    
        if U_tree[0] == (h, u) :
            return U_tree
        else:
            desc_1 = find_desc(U_tree[1], (h, u))
            if desc_1 != None:
                return desc_1
            desc_2 = find_desc(U_tree[2], (h, u))
            if desc_2 != None:
                return desc_2
            desc_3 = find_desc(U_tree[3], (h, u))
            if desc_3 != None:
                return desc_3
            desc_4 = find_desc(U_tree[4], (h, u))
            if desc_4 != None:
                return desc_4


# In[17]:


print find_desc(U_tree, (4, (2, 3)))


# In[18]:


def calcule_asc_path(h,u):
    rep = []
    for h_inf in range(h, 0, -1):
        i_inf = u[0] / (2 ** (h - h_inf))
        j_inf = u[1] / (2 ** (h - h_inf))
        rep += [(h_inf, (i_inf, j_inf))]
    # racine
    rep += [(0, (i_inf, j_inf))]
    return rep


# In[19]:


print calcule_asc_path(5,(15,3))


# #### Etapes de calcul du posterior : likelihood - log-score - posterior

# In[20]:


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


# In[21]:


# Log posterior

def update_log_score(log_score, lik):
    #print 'lik =' + str(lik) 
    log_score += np.log(lik) 
    max_log_score = max(log_score)
    log_score -= max_log_score
    return log_score


# In[22]:


# Posterior (Softmax)
    
def calc_pi(log_score): # TODO
    Z = np.sum(np.exp(log_score))
    pi = np.exp(log_score)/Z
    #print 'pi =' + str(pi)
    #print 'max(pi) = ',max(pi)
    return pi


# #### Métriques

# In[23]:


# Entropy (over counterfactual viewpoint (h_plus, u_plus))

def calc_H_plus(log_score, v_plus, h_plus, u_plus):
    lik_plus = calc_lik(v_plus, h_plus, u_plus)
    log_score_plus = update_log_score(np.copy(log_score), lik_plus)
    pi_plus = calc_pi(log_score_plus)
    sum_H = - pi_plus * np.log(pi_plus)
    #print 'sum_H : ', sum_H
    #print np.log(lik)
    #print '****', 'c :', c_ref, ', h :', h, ', u :',u, ' ---> ', np.where(pi == max(pi))[0][0]
    #print '**** H = ',
    return sum_H.sum()


# In[24]:


# Free Energy (over counterfactual viewpoint (h_plus, u_plus))

def calc_F_plus(pi, log_score, v_plus, h_plus, u_plus):    
    lik_plus = calc_lik(v_plus, h_plus, u_plus)
    log_score_plus = update_log_score(np.copy(log_score), lik_plus)
    pi_plus = calc_pi(log_score_plus)
    sum_F = - pi_plus * (np.log(lik_plus) - np.log(pi_plus) + np.log(pi))
    #print 'sum_F : ', sum_F
    #print np.log(lik)
    #print '****', 'c :', c_ref, ', h :', h, ', u :',u, ' ---> ', np.where(pi == max(pi))[0][0]
    #print '**** H = ',
    return sum_F.sum()   


# In[25]:


def calc_IB_plus(pi, log_score, v_plus, h_plus, u_plus):    
    lik_plus = calc_lik(v_plus, h_plus, u_plus)
    log_score_plus = update_log_score(np.copy(log_score), lik_plus)
    pi_plus = calc_pi(log_score_plus)
    sum_IB = pi_plus * (np.log(pi_plus) - np.log(pi))
    #print 'sum_F : ', sum_F
    #print np.log(lik)
    #print '****', 'c :', c_ref, ', h :', h, ', u :',u, ' ---> ', np.where(pi == max(pi))[0][0]
    #print '**** H = ',
    return sum_IB.sum()   


# In[26]:


# Gibbs Energy

def calc_E_plus(pi, v_plus, h_plus, u_plus):    
    lik_plus = calc_lik(v_plus, h_plus, u_plus)
    sum_E = pi * lik_plus
    #print 'sum_F : ', sum_F
    #print np.log(lik)
    #print '****', 'c :', c_ref, ', h :', h, ', u :',u, ' ---> ', np.where(pi == max(pi))[0][0]
    #print '**** H = ',
    return sum_E.sum()   


# #### Generators

# In[27]:


def argmax_generator(c, h, u):
    test_pred = rho[c][h][u] < .5       
    if test_pred:
        return mu[c][h][u]
        #v_predictive = np.random.multivariate_normal(mu[c_predictive][h_path][u_path], Sigma[c_predictive][h_path][u_path], 1)[0]
    else:
        return np.zeros(3)


# In[28]:


def monte_carlo_generator(c, h, u):
    if np.random.random() > rho[c][h][u]:
        return np.random.multivariate_normal(mu[c][h][u], Sigma[c][h][u], 1)[0]
    else:
        return np.zeros(3)


# #### Likelihood map : `lik_predictive[c][h][u]`

# In[30]:


h_max = 6
lik_predictive = {}
for c in range(10):
    lik_predictive[c] = {}
    for h in range(h_max):
        lik_predictive[c][h] = {}
        for u in U[h]:
            log_score = np.zeros(10)
            v_predictive = argmax_generator(c, h, u)
            lik = calc_lik(v_predictive, h, u)
            lik_predictive[c][h][u] = lik


# #### Effective saliency map : `pi_predictive_eff[c][u]` (h = 5) 

# In[32]:


if not os.path.isfile("mnist-waveimage-saliency-map.pkl"):
    h = h_max - 1
    pi_predictive_eff = {}
    for c in range(10):
        print c
        pi_predictive_eff[c] = {}

        mem_h_u = []
        mem_h_u_todo = {}
        for u_add in U[h]:
            mem_h_u_todo[(5, u_add)] = 1

        while len(mem_h_u_todo) > 0 :    
            pi_predictive_plus = {}
            for (h, u) in mem_h_u_todo:
                liste_path = calcule_asc_path(h, u)
                log_score_path = np.zeros(10)
                for (h_path, u_path) in liste_path[:-1]:
                    if (h_path, u_path) not in mem_h_u:
                        log_score_path = update_log_score(log_score_path, lik_predictive[c][h_path][u_path])
                pi_path = calc_pi(log_score_path)
                #print pi_path
                pi_predictive_plus[u] = pi_path[c] 
                #print h_plus, u_plus
            keys = pi_predictive_plus.keys()
            values = np.array(pi_predictive_plus.values())
            k = np.argmax(values)
            u_tilde = keys[k]
            #print len(pi_predictive_plus), len(mem_h_u), (values[k], (h, u))

            pi_predictive_eff[c][u_tilde] = values[k]
            mem_h_u_todo.pop((h, u_tilde)) 
            liste_path = calcule_asc_path(h, u_tilde)
            log_score_path = np.zeros(10)
            for (h_path, u_path) in liste_path[:-1]:
                if (h_path, u_path) not in mem_h_u:
                    mem_h_u += [(h_path, u_path)]       
    pickle.dump(pi_predictive_eff,  open("mnist-waveimage-saliency-map.pkl", "wb"))
else:
    pi_predictive_eff = pickle.load(open("mnist-waveimage-saliency-map.pkl", "rb"))            
        #print 'CHOIX :', (h, u)
        
        
        


# #### Generic saliency map : `H_generic_eff[c][u]` (h = 5) 

# In[31]:


from scipy.stats import entropy


# In[34]:


if not os.path.isfile("mnist-waveimage-generic-saliency-map.pkl"):
    h = h_max - 1
    H_generic_eff = {}

    mem_h_u = []
    mem_h_u_todo = {}
    for u_add in U[h]:
        mem_h_u_todo[(5, u_add)] = 1
    cpt = 0
    while len(mem_h_u_todo) > 0 :    
        cpt += 1
        H_predictive_plus = {}
        for (h, u) in mem_h_u_todo:
            H_predictive_plus[u] = 0
        for (h, u) in mem_h_u_todo:
            liste_path = calcule_asc_path(h, u)
            for c in range(10):
                log_score_path = np.zeros(10)
                for (h_path, u_path) in liste_path[:-1]:
                    if (h_path, u_path) not in mem_h_u:
                        log_score_path = update_log_score(log_score_path, lik_predictive[c][h_path][u_path])
                pi_path = calc_pi(log_score_path)
                H_c = entropy(pi_path)
                H_predictive_plus[u] += .1 * H_c
        keys = H_predictive_plus.keys()
        values = np.array(H_predictive_plus.values())
        k = np.argmin(values)
        u_tilde = keys[k]
        print(cpt, u_tilde)
        H_generic_eff[u_tilde] = values[k]
        mem_h_u_todo.pop((h, u_tilde)) 
        liste_path = calcule_asc_path(h, u_tilde)
        for (h_path, u_path) in liste_path[:-1]:
            if (h_path, u_path) not in mem_h_u:
                mem_h_u += [(h_path, u_path)]       
    pickle.dump(H_generic_eff,  open("mnist-waveimage-generic-saliency-map.pkl", "wb"))
else:
    H_generic_eff = pickle.load(open("mnist-waveimage-generic-saliency-map.pkl", "rb"))            
        #print 'CHOIX :', (h, u)
        
        
        


# In[59]:


## test 
h = h_max - 1
H_generic = {}

mem_h_u = []
mem_h_u_todo = {}
for u_add in U[h]:
    mem_h_u_todo[(5, u_add)] = 1
cpt = 0
while len(mem_h_u_todo) > 0 :    
    cpt += 1
    H_predictive_plus = {}
    for (h, u) in mem_h_u_todo:
        H_predictive_plus[u] = 0
    for (h, u) in mem_h_u_todo:
        liste_path = calcule_asc_path(h, u)
        for c in range(10):
            log_score_path = np.zeros(10)
            for (h_path, u_path) in liste_path[:-1]:
                if (h_path, u_path) not in mem_h_u:
                    log_score_path = update_log_score(log_score_path, lik_predictive[c][h_path][u_path])
            pi_path = calc_pi(log_score_path)
            H_c = entropy(pi_path)
            H_predictive_plus[u] += .1 * H_c
    keys = H_predictive_plus.keys()
    values = np.array(H_predictive_plus.values())
    k = np.argmin(values)
    u_tilde = keys[k]
    print(cpt, u_tilde)
    H_generic[u_tilde] = values[k]
    mem_h_u_todo.pop((h, u_tilde)) 

        


# In[36]:


H_generic_eff


# ### Reconstruction

# In[37]:


avg_image = {}
for c in range(10):
    w = WaveImage()
    for h in range(6):
        for u in calc_U((32, 32), h, 6):
            w.set_data( h, u, mu[c][h][u] * (1 - rho[c][h][u]))
    avg_image[c] = w.get_image()
    plt.figure(figsize=(12,8))
    plt.subplot(131)
    plt.imshow(w.get_image(), interpolation='nearest', cmap='gray_r')



# In[38]:


def calc_pi_predictive_sorted(pi_predictive): 
    pi_predictive_sorted = {}
    for c in range(10):
        pi_predictive_sorted[c] = []
        for k in pi_predictive[c]:
            pi_predictive_sorted[c] += [(pi_predictive[c][k], k)]
        pi_predictive_sorted[c] = sorted(pi_predictive_sorted[c])
    return pi_predictive_sorted


# In[54]:


def calc_H_predictive_sorted(H_predictive): 
    H_predictive_sorted = []
    for k in H_predictive:
        H_predictive_sorted += [(H_predictive[k], k)]
    H_predictive_sorted = sorted(H_predictive_sorted, reverse=True)
    return H_predictive_sorted


# In[39]:


def affiche_path_mnist(path_i, path_j):
    col_max = max(15, len(path_i))
    colors = plt.cm.rainbow(np.linspace(0, 1, col_max))
    b_moins = -.5
    b_plus = 31.5
    for cpt in range(len(path_i) - 1):
        plt.plot(path_j[cpt:cpt + 2],path_i[cpt:cpt + 2], color = colors[col_max - cpt - 1], linewidth= 3)
    #plt.plot(path_j,path_i,'r+',markersize=12)
    plt.xlim([b_moins,b_plus])
    plt.ylim([b_moins,b_plus])
    plt.gca().invert_yaxis()


# In[40]:


def affiche_path(pi_predictive): 
    pi_predictive_sorted = calc_pi_predictive_sorted(pi_predictive)
    for c in range(10):
        image_moy = avg_image[c] #np.mean(np.array(data_visu[c]),0)
        plt.figure(figsize=(12,8))
        plt.subplot(131)
        plt.imshow(image_moy, interpolation='nearest', cmap='gray_r')
        plt.title(c)
        path_i = []
        path_j = []
        image_pi = np.zeros((14,14))
        for cpt in range(6):
            (pi_pred, u) = pi_predictive_sorted[c][-1-cpt]
            #(pi_pred,(h,u)) = pi_predictive_eff[c][cpt]
            #print c,pi_pred
            path_i += [u[0] * 2 + 1.5]
            path_j += [u[1] * 2 + 1.5]
        affiche_path_mnist(path_i, path_j)
        plt.plot(path_j[0], path_i[0],'+r', markersize = 15, mew = 3)

        plt.subplot(132)
        sal_map = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                sal_map[i,j] = pi_predictive[c][(i,j)] 



        plt.imshow(sal_map, interpolation='nearest', cmap = 'gist_heat_r', vmin = 0, vmax = 1)


# In[57]:


def affiche_path_H(H_predictive): 
    H_predictive_sorted = calc_H_predictive_sorted(H_predictive)
    #image_moy = avg_image[c] #np.mean(np.array(data_visu[c]),0)
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    #plt.imshow(image_moy, interpolation='nearest', cmap='gray_r')
    #plt.title(c)
    path_i = []
    path_j = []
    image_pi = np.zeros((14,14))
    for cpt in range(6):
        (H_pred, u) = H_predictive_sorted[-1-cpt]
        #(pi_pred,(h,u)) = pi_predictive_eff[c][cpt]
        #print c,pi_pred
        path_i += [u[0] * 2 + 1.5]
        path_j += [u[1] * 2 + 1.5]
    affiche_path_mnist(path_i, path_j)
    plt.plot(path_j[0], path_i[0],'+r', markersize = 15, mew = 3)

    plt.subplot(132)
    sal_map = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            sal_map[i,j] = H_predictive[(i,j)] 

    plt.imshow(sal_map, interpolation='nearest', cmap = 'gist_heat_r')


# In[60]:


affiche_path_H(H_generic)


# In[61]:


affiche_path_H(H_generic_eff)

