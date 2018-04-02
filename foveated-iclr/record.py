import numpy as np
import matplotlib.pyplot as plt

class Record:
    def __init__(self):
        self.POL = None
        self.THRESHOLD = None
        self.nb_saccades = 0
        self.nb_coeffs = 0
        self.mem_u = []
        self.mem_h_u = []
        self.mem_pi = []
        self.mem_H = []
        self.mem_z = []
        self.z_ref = None
        self.z_final = None
        self.success = None
    def __str__(self):
        return  'POL : ' + str(self.POL) + '\n' +\
                'THRESHOLD : ' + str(self.THRESHOLD) + '\n' +\
                'nb_saccades : ' + str(self.nb_saccades) + '\n' +\
                'nb_coeffs : ' + str(self.nb_coeffs) + '\n' +\
                'mem_u : ' + str(self.mem_u) + '\n' +\
                'mem_h_u : ' + str(self.mem_h_u) + '\n' +\
                'mem_pi : ' + str(self.mem_pi) + '\n' +\
                'mem_H : ' + str(self.mem_H) + '\n' +\
                'mem_z : ' + str(self.mem_z) + '\n' +\
                'z_ref : ' + str(self.z_ref) + '\n' +\
                'z_final : ' + str(self.z_final) + '\n' +\
                'success : ' + str(self.success)
  
# Records list
def affiche_records(records, liste_NB_TRIALS):
    
    NB_CLASSES = len(liste_NB_TRIALS)
    
    mem_classif = []
    mem_saccades = []
    mem_coeffs = []
    mem_compression_rate = []
    mem_posterior = []
    mem_entropy = []
    POL = records[0].POL
    THRESHOLD = records[0].THRESHOLD

    cpt_TRIALS = 0
    for z_ref in range(NB_CLASSES):
        NB_TRIALS = liste_NB_TRIALS[z_ref] #len(Data_test[z_ref][0][(0,0)])
        for ind_test in range(NB_TRIALS):
            ind_total = cpt_TRIALS + ind_test
            mem_classif += [records[ind_total].success]
            mem_saccades += [records[ind_total].nb_saccades]
            mem_coeffs += [records[ind_total].nb_coeffs]
            compression_rate = 100 * (1 - records[ind_total].nb_coeffs / 28. / 28.)  
            mem_compression_rate += [compression_rate]
            pi = records[ind_total].mem_pi[-1]
            mem_posterior += [pi[records[ind_total].z_final]] 
            mem_entropy += [records[ind_total].mem_H[-1]]
        cpt_TRIALS += NB_TRIALS
    
    print '\n'
    print 'Policy :', POL
    print 'Threshold :', THRESHOLD
    classif_rate = np.mean(np.where(np.array(mem_classif)==True,1,0)) * 100.0
    print 'classif rate :', classif_rate
    print 'nb_saccades :', np.mean(mem_saccades)  
    print 'nb_coeffs :', np.mean(mem_coeffs)  
    #compression_rate = 100 * (1 - np.mean(mem_coeffs) / 28 / 28)  
    print 'compression rate :',  np.mean(mem_compression_rate)
    print 'final posterior :', np.mean(mem_posterior)
    print 'final entropy :', np.mean(mem_entropy)

    #plt.plot(mem_entropy)
    plt.figure()
    plt.hist(mem_saccades, np.arange(0, 270, 4))
    plt.yticks([])
    plt.xlim([0,260])
    plt.xlabel('# saccades')
    plt.title(POL + ' policy, $H_{ref}$ = ' + str(THRESHOLD))
    a = plt.axes([.35, .5, .5, .3])
    n, bins, patches = plt.hist(mem_saccades, np.arange(257))
    plt.xlabel('# saccades')
    plt.yticks([])
    plt.xlim([0,31])
    
    return classif_rate, mem_saccades, mem_compression_rate
    