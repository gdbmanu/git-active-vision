import numpy as np
import scipy as sp
import pywt
import math

# Tree strcture representation of an image
# intended to be used on the MNIST database

def calc_dim(shape, h, h_max):
	assert 0 <= h < h_max
	if h == 0:
		dim_i = int(math.ceil(shape[0] * 1. / 2**(h_max - 1)))
		dim_j = int(math.ceil(shape[1] * 1. / 2**(h_max - 1)))
	else :
		dim_i = int(math.ceil(shape[0] * 1. / 2**(h_max - h)))
		dim_j = int(math.ceil(shape[1] * 1. / 2**(h_max - h)))
	return dim_i, dim_j

def calc_U(shape, h, h_max): #dim_i, dim_j):
    dim_i, dim_j = calc_dim(shape, h, h_max)
    U = []
    for i in range(dim_i):
        for j in range(dim_j):
            U += [(i, j)]
    return U

def mnist_reshape_32(x, i_offset = 0, j_offset = 0):
    assert x.shape == (28 * 28,)
    image = x.reshape(28, 28)
    image = np.append(np.zeros((16 + 2, 28)), image, axis = 0)
    image = np.append(image, np.zeros((16 + 2, 28)), axis = 0)
    image = np.append(np.zeros((32 + 32, 16 + 2)), image, axis = 1)
    image = np.append(image, np.zeros((32 + 32, 16 + 2)), axis = 1)
    return image[16 + i_offset : 48 + i_offset, 16 + j_offset : 48 + j_offset]

class WaveImage:
	
	def __init__(self, image = None, shape = (32, 32)):
		
		# Attribut shape
		if image is not None:
			# Decomposition ondelettes
			coeffs = pywt.wavedec2(image, 'haar')
			self.__shape = image.shape
		else:
			self.__shape = shape		
		
		# Attribut h_max : profondeur de l'image
		self.__h_max = min(int(math.log(self.__shape[0], 2)) + 1, 	int(math.log(self.__shape[1], 2)) + 1)
			
		# Attribut data : L'attribut data contient les vecteurs en position [h][u] (dictionnaire)
		if image is not None:
			self.__data = {}
			for h in range(self.__h_max):
				self.__data[h] = {}
				if h == 0:
					(i_max, j_max) = coeffs[h].shape
				else:
					(i_max, j_max) = coeffs[h][0].shape
				for i in range(i_max):
					for j in range(j_max):
						if h == 0:
							data = coeffs[h][i][j]
						else:
							data = coeffs[h][0][i][j]
							for k in range(1,len(coeffs[h])):
								data = np.append(data, coeffs[h][k][i][j])	
						self.__data[h][(i, j)] = data				
		else: # image is None
			self.__data = {}
			for h in range(self.__h_max):
				self.__data[h] = {}
					
		
	def get_data(self):
		return self.__data
	
	def get_shape(self):
		return self.__data
				
	def set_data(self, h, u, v):
		assert 0 <= h < self.__h_max
		dim_i, dim_j = calc_dim(self.__shape, h, self.__h_max)
		assert 0 <= u[0] < dim_i
		assert 0 <= u[1] < dim_j
		if h == 0 :
			self.__data[h][u] = v
		else:
			self.__data[h][u] = np.copy(v)
		
	def get_h_max(self):
		return self.__h_max
		
	def get_image(self):
		coeffs = []
		for h in range(self.__h_max):
			dim_i, dim_j = calc_dim(self.__shape, h, self.__h_max)
			if h == 0:
				coeffs_h = np.zeros((dim_i, dim_j))
				for u in self.__data[h]:
					coeffs_h[u[0],u[1]] = self.__data[h][u]
			else:
				coeffs_h = [np.zeros((dim_i, dim_j)), np.zeros((dim_i, dim_j)), np.zeros((dim_i, dim_j))]
				for u in self.__data[h]:
					for k in range(3):
						coeffs_h[k][u[0],u[1]] = self.__data[h][u][k]
			coeffs += [coeffs_h]
		return pywt.waverec2(coeffs, 'haar')	
		
	def add_coeffs(self, waveImage, u, h_ref = 0):
		# Niveau 0
		h_opp = self.__h_max - 1
		i = int(u[0] / 2**h_opp) 
		j = int(u[1] / 2**h_opp)
		u_0 = (i,j)
		if self.__data[0] == {}:
			self.__data[0][u_0] = waveImage.get_data()[0][u_0]
		else:
			v_test = self.__data[0][u_0]
			if np.linalg.norm(v_test) < 1e-16:
				self.__data[0][u_0] = waveImage.getData()[0][u_0]
		# Niveaux 1 et +
		for h in range(1, h_ref) :
			h_opp = self.__h_max - h
			i = int(u[0] / 2**h_opp) 
			j = int(u[1] / 2**h_opp)
			if (i,j) in self.__data[h]:
				v_test = self.__data[h][(i,j)]
				if np.linalg.norm(v_test) < 1e-16:
					self.__data[h][(i,j)] = np.copy(waveImage.get_data()[h][(i,j)])
			else: 
				self.__data[h][(i,j)] = np.copy(waveImage.get_data()[h][(i,j)])
	
	def copy(self):
		self_shape = self.__shape 
		self_copy = WaveImage(shape = self_shape)
		for h in range(self.__h_max) :
			for u in self.__data[h]:
				self_copy.set_data(h, u, self.__data[h][u])
		return self_copy	
		
	def __str__(self):
		h_max = len(self.__data)
		s = 'h_max :' + str(self.__h_max) + '\n'
		for h in range(self.__h_max):
			s += '***' + str(h) + '***\n'
			s += str(self.__data[h]) + '\n'
		return s

class WaveDict:
	def __init__(self, shape = (32, 32), nb_classes = 10):
		# Attribut shape
		self.__shape = shape	
		# Attribut h_max : profondeur de l'image
		self.__h_max = min(int(math.log(self.__shape[0], 2)) + 1, 	int(math.log(self.__shape[1], 2)) + 1)
		# Attribut data : L'attribut data contient les vecteurs en position [c][h][u] (dictionnaire)
		self.__data = {}
		for c in range(nb_classes):
			self.__data[c] = {}
			for h in range(self.__h_max):
				self.__data[c][h] = {}
				dim_i, dim_j = calc_dim(self.__shape, h, self.__h_max)
				for i in range(dim_i):
					for j in range(dim_j):
						self.__data[c][h][(i, j)] = []
		self.__nb_classes = nb_classes
						
	def get_shape(self):
		return self.__shape
	
	def get_h_max(self):
		return self.__h_max
		
	def get_data(self):
		return self.__data
		
	def add(c, h, u, v):
		assert 0 <= c < self.__nb_classes
		assert 0 <= h < self.__h_max
		if u in Data[c][h]:
			self.__data[c][h][u] += [v]
		else:
			self.__data[c][h][u] = [v]
				
		
	
