'''
******
 PCML 13 - Miniproject
 Part I
 Learning a Multilayer Perceptron

 Problems:
 1) distinguish between digits 3 and 5
 2) distinguish between digits 4 and 9

 digit resolution: 28x28 stored as vector of length 784

******
'''

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math
import random
import time

class MultiLayerPerceptron:

	def __init__(self, h1, X, Y, X_valid, Y_valid):
		self.feature_dim = X.shape[1] 
		self.h1 = h1
		self.num_as = 2*h1

		'''
			assign data 
		'''
		self.X = X
		self.Y = Y
		self.X_valid = X_valid
		self.Y_valid = Y_valid
		
		'''
		 	initialize weights
		'''

		# w1s Nact x feature_dim 
		# remove the for, add a coutner
		self.w1s = np.array([[np.random.normal(0,1.0/self.feature_dim) for i in range(self.feature_dim)] for i in range(self.num_as)])
		self.w2 = np.array([np.random.normal(0,1.0/self.h1) for i in range(0,self.h1)])

		#print "w1s: "+str(w1s)
		#print "w2: "+str(w2)

		'''
			initialize biases
		'''
		# as many bias terms as as outputs in hidden layer
		self.b1 = np.array([np.random.normal(.5,.25) for i in range(0,self.num_as)])
		self.b2 = np.random.normal(.5,.25)

		'''
			initialize gradients, learning rate and momentum term
		'''
		self.dgw1 = np.zeros(self.w1s.shape)
		self.dgw2 = np.zeros(self.w2.shape)
		self.dgb1 = np.zeros(self.b1.shape)
		self.dgb2 = 0
		self.eta = 0.01
		self.mu = 0.2

		'''
			gdescent loop variables, errors
		'''

		self.epochs = 0
		self.max_epochs = 50000 
		self.num_iter = 0 

		self.cvgc = 1E-8
		self.cvg = False

		self.train_err_arr = np.empty(0)
		self.valid_err = 1E6
		self.valid_err_arr = np.empty(0)
		self.zerone = np.empty(0)

	'''
		sigmoid function
		parameter: vector of activation values
	'''
	def sigmoidf(self, x):
		expx = np.exp(-x)	
		return 1./(1.+expx)

	'''
		gating function for pairs of activation values (vectors)
		parameters: a2k (vector), a2kp1 (vector)
	'''
	def gatingf(self, a2km1, a2k):
		return a2km1*self.sigmoidf(a2k)

	'''
		run forward pass in batch_mode to compute a(2) for full error
	'''
	def forward_prop_batch(self, curr_X):

		curr_num_points = curr_X.shape[0]
		# dot() - for 2D arrays equivalent to matrix multiplication!
		self.a1s = np.dot(self.w1s,curr_X.T)+np.tile(self.b1, (curr_num_points, 1)).T
		#print self.a1s
		#a11 = np.dot(self.w1s[0],x[0])
		#print a11

		# vectorize
		a2km1 = self.a1s[0:self.num_as:2, :]
		#print "a2k ",a2km1
		a2k = self.a1s[1:self.num_as:2, :]
		#print "a2kp1 ",a2k

		# np.apply_along_axis for multiple inputs?
		z = np.array([self.gatingf(a2km1[:,i], a2k[:,i]) for i in range(curr_num_points)])
		#print "z ",z.shape
		#print self.w2.shape
		#print "dot z self.w2",np.dot(z, self.w2)
		
		self.a2 = np.dot(z,self.w2)+np.tile(self.b2, (curr_num_points,))
		#print "shape a2 ", self.a2.shape
		#print "a2 ",self.a2
		return

	'''
		run forward pass online for datapoint x_i for stochastic gradient descent
		NOTE: pass index instead of data
	'''
	def forward_prop_online(self, i):
		# dot() - for 2D arrays equivalent to matrix multiplication!
		#if self.epochs == 0:
		x = self.X[i]
		self.a1s = np.dot(self.w1s,x)+self.b1
		#print self.a1s
		a11 = np.dot(self.w1s[0],x)
		#print a11

		# vectorize
		a2km1 = self.a1s[0:self.num_as:2]
		a2k = self.a1s[1:self.num_as:2]

		self.z = self.gatingf(a2km1, a2k)
		#print self.z

		self.a2 = np.dot(self.w2, self.z)+self.b2
		#print "a2 ",self.a2
		'''	
		else: 
			# just use previously computed a2 value for index i in total error
			self.z = self.z[i]
			self.a2 = self.a2[i]
		'''

		return

	'''
		compute residuals for logarithmic error function online_mode --> stochastic gradient descent
	'''
	def log_res(self, label):
		# labels - N x 1
		dlabel = (label+1)/2
		self.r2 = self.sigmoidf(self.a2)-dlabel
		#self.r2 = -label*np.exp(-label*self.a2)*self.sigmoidf(label*self.a2)
		#rks_sigm = np.array([[self.w1s[i][k]*self.sigmoidf(self.a1s[i][k]) if (k%2==0) else self.w1s[i][k]*self.sigmoidf(self.a1s[i][k])*self.sigmoidf(-self.a1s[i][k]) for k in range(0,self.num_as)] for i in range(0, self.feature_dim)])
		#r1s_sigm = np.array([self.w2[k/2]*self.sigmoidf(self.a1s[k+1]) if (k%2==0) else self.w2[k/2]*self.a1s[k-1]*np.exp(-self.a1s[(k+1)/2])*self.sigmoidf(self.a1s[(k+1)/2])*self.sigmoidf(self.a1s[(k+1)/2]) for k in range(self.num_as)])
		r1s_sigm = np.array([self.w2[k/2]*self.sigmoidf(self.a1s[k]) if (k%2==0) else self.w2[k/2]*self.a1s[k-1]*self.sigmoidf(self.a1s[k])*self.sigmoidf(-self.a1s[k]) for k in range(self.num_as)])

		self.r1s = self.r2*r1s_sigm 
		
		return

	def back_prop_online(self, i):
		label = self.Y[i]
		self.log_res(label)
		x = self.X[i]
		# assemble gradient
		self.gw2 = self.r2*self.z
		#print "gw2 shape: ",gw2.shape
		self.gb2 = self.r2
		#print "gb2 shape: ",gb2.shape
		self.gw1 = np.dot(np.tile(x, (self.num_as,1)).T, np.diag(self.r1s)).T

		self.gb1 = self.r1s 
		#print "gb1 shape: ",gw1.shape

		#self.dgw2 = -self.eta*self.gw2
		self.dgw2 = -self.eta*(1-self.mu)*self.gw2+self.mu*self.dgw2
		self.w2 = self.w2+self.dgw2

		#self.dgw1 = -self.eta*self.gw1
		self.dgw1 = -self.eta*(1-self.mu)*self.gw1+self.mu*self.dgw1
		self.w1s = self.w1s+self.dgw1

		self.dgb2 = -self.eta*(1-self.mu)*self.gb2+self.mu*self.dgb2
		#self.dgb2 = -self.eta*self.gb2
		self.b2 = self.b2+self.dgb2
		
		self.dgb1 = -self.eta*(1-self.mu)*self.gb1+self.mu*self.dgb1
		#self.dgb1 = -self.eta*self.gb1
		self.b1 = self.b1+self.dgb1

		return

	def eval_err(self, X, Y):
		'''
			This is working
		'''
		# forward pass for all
		self.forward_prop_batch(X)
		output = Y.flatten()*self.a2
		
		curr_num_points = X.shape[0]

		pos_ind = np.where(output > 0)[0]
		cor_class = len(pos_ind)

		neg_ind = np.where(output <= 0)[0]
		zerone_err = np.sum(neg_ind)*1.0/curr_num_points
		mis_class = len(neg_ind)

		pos_output = np.zeros(output.shape[0])
		neg_output = np.zeros(output.shape[0])

		pos_output[pos_ind] = output[pos_ind]
		neg_output[neg_ind] = output[neg_ind]

		pos_ind_err = np.log(1.0+np.exp(-pos_output))
		pos_ind_err[neg_ind] = 0

		neg_ind_err = -neg_output+np.log(1.0+np.exp(neg_output))
		neg_ind_err[pos_ind] = 0

		log_err_is = pos_ind_err + neg_ind_err
		log_err = np.sum(log_err_is)/curr_num_points

		return (log_err, mis_class, cor_class, zerone_err)

	def eval_train_err(self):
		ret = self.eval_err(self.X, self.Y)
		self.train_err = ret[0]
		self.train_mis_class = ret[1]*100.0/(self.X_valid.shape[0])
		self.train_pos_class = ret[2]*100.0/(self.X_valid.shape[0])
		self.train_err_arr = np.append(self.train_err_arr, self.train_err)
		return

	def eval_valid_err(self):
		self.prev_valid_err = self.valid_err
		ret = self.eval_err(self.X_valid, self.Y_valid)
		self.valid_err = ret[0]
		self.valid_mis_class = ret[1]*100.0/self.X_valid.shape[0]
		self.valid_cor_class = ret[2]*100.0/self.X_valid.shape[0]
		zerone_err = ret[3]
		self.zerone = np.append(self.zerone, zerone_err)
		self.valid_err_arr = np.append(self.valid_err_arr, self.valid_err)
		return

	def print_status(self):
		print "train_err" ,self.train_err
		print "valid_err", self.valid_err
		return

	def disp_results(self):
		print "Results validation:"
		print "perc. correct classifications: ",self.valid_cor_class
		print "perc. misclassifications: ",self.valid_mis_class

		X = np.linspace(1, self.epochs, self.epochs, endpoint=True)

		plt.clf()
		plt.plot(X, self.train_err_arr, color="blue", linewidth=1.0, linestyle='-', label="training error")
		plt.plot(X, self.valid_err_arr, color="red", linewidth=1.0, linestyle='-', label="validation error")

		plt.xlim(1, self.epochs)
		plt.ylabel('error')
		plt.xlabel('epochs')
		plt.legend(loc='upper right')
		plt.savefig('plots/mlp_plot.png')
		plt.show()

		print self.zerone
		plt.clf()		
		plt.plot(X, self.zerone, color="purple", linewidth=1.0, linestyle='-', label="zero one error")
		plt.xlim(1, self.epochs)
		plt.ylim(np.min(self.zerone)-10, np.max(self.zerone)+10)
		plt.ylabel('zerone error')
		plt.xlabel('epochs')
		plt.legend(loc='upper right')
		plt.show()

		return


	def gdescent(self):

		fig = plt.figure(1)
		while not self.cvg:

			rand_perm = np.random.permutation(self.X.shape[0])

			for i in rand_perm:
				'''
				print self.w1s
				print self.w2
				print self.b1
				print self.b2
				'''
				self.num_iter+=1
				self.forward_prop_online(i)
				self.back_prop_online(i)

			self.eval_train_err() # includes batch forward prop
			self.eval_valid_err()
			self.print_status()

			self.epochs+=1
			self.eta-=1./self.X[0].shape[0]

			if self.valid_err > self.prev_valid_err:
				print '*** Convergence after ', self.epochs, ' epochs ***'	
				self.disp_results()
				self.cvg = True

			if self.epochs >= self.max_epochs:
				print '*** Reached max. num of epochs', self.max_epochs, ' ***'
				self.cvg = True


		return

	# split data into TRAINING set (2/3) and VALIDATION set (1/3); TEST set is provided separately

	# PREPROCESSING DATA: normalize input patterns to have coefficients in [0,1]
	# get max coefficient and min coefficient

	# scale and center features

	'''
		stochastic gradient descent
		- pick training case at random i(k) - compute gradient gk=gradient(wk)Ei(k)
		- update wk+1 = wk - thetak*gk
	'''

def main():

	'''
	train_data = np.array([[0,1], [0,0], [1,0], [1,1]]) 
	print train_data.shape
	train_labels = np.array([1,-1,1,-1])

	valid_data = train_data
	valid_labels = train_labels 

	h1 = 4 

	xormlp = MultiLayerPerceptron(h1, train_data, train_labels, valid_data, valid_labels)
	xormlp.gdescent()

	'''
	#train_data = np.load('50_training_data.npy')
	#train_labels = np.load('50_training_labels.npy')
	train_data = np.load('training_data.npy')
	train_labels = np.load('training_labels.npy')
	#valid_data = np.load('16_validation_data.npy')
	#valid_labels = np.load('16_validation_labels.npy')
	#valid_data = np.load('50_validation_data.npy')
	#valid_labels = np.load('50_validation_labels.npy')
	valid_data = np.load('validation_data.npy')
	valid_labels = np.load('validation_labels.npy')

	test_data = np.load('test_data.npy')
	test_labels = np.load('test_labels.npy')

	print train_data 
	print len(train_data)
	print train_labels

	h1 = 65 
	mlp = MultiLayerPerceptron(h1, train_data, train_labels, valid_data, valid_labels)
	mlp.gdescent()
	result = mlp.eval_err(test_data, test_labels)
	print result

	''' record
	train_err 0.0365741141353
	valid_err 0.0619260675686
	*** Convergence after  5  epochs ***
	train_err 0.0255103947063
	valid_err 0.0532923952223
	test_err 0.055956285193
	*** Convergence after  6  epochs ***
	train_err 0.032744899583
valid_err 0.0629107588134
*** Convergence after  6  epochs ***
train_err 0.0334248445863
valid_err 0.0561047951724
*** Convergence after  6  epochs ***
Results validation:
perc. correct classifications:  97.95


	'''

	return

if __name__ == "__main__":
	main()		

