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

import random

class MultiLayerPerceptron:

	def __init__(self, h1, X, Y):
		self.feature_dim = X.shape[1] 
		self.h1 = h1
		self.num_as = 2*h1
		# initialize w's
		# w1s Nact x feature_dim 
		self.w1s = np.array([[np.random.normal(0,1.0/self.feature_dim) for i in range(self.feature_dim)] for i in range(self.num_as)])
		#print "w1s: "+str(w1s)
		self.w2 = np.array([np.random.normal(0,1.0/self.h1) for i in range(0,self.h1)])
		#print "w2: "+str(w2)
		# as many bias terms as as outputs in hidden layer
		self.b1 = np.array([np.random.normal(.5,.25) for i in range(0,self.num_as)])
		self.b2 = np.random.normal(.5,.25)
		self.X = X
		self.num_points = len(X)
		self.Y = Y
		self.epochs = 0
		self.max_epochs = 10000
		self.cvgc = 0.00001
		self.cvg = False
	'''
		sigmoid function
		parameter: vector of activation values
	'''
	def sigmoidf(self, x):
		expx = np.exp(x)	
		return 1/(1+expx)

	'''
		gating function for pairs of activation values (vectors)
		parameters: a2k (vector), a2kp1 (vector)
	'''
	def gatingf(self, a2k, a2kp1):
		return a2k*self.sigmoidf(a2kp1)

	'''
		run forward pass in batch_mode to compute a(2) for full error
	'''
	def forward_prop_batch(self):
		# dot() - for 2D arrays equivalent to matrix multiplication!
		self.a1s = np.dot(self.w1s,self.X.T)+self.b1
		print self.a1s
		#a11 = np.dot(self.w1s[0],x[0])
		print a11

		# vectorize
		a2k = self.a1s[:,0:self.num_as:2]
		print a2k
		a2kp1 = self.a1s[:,1:self.num_as:2]
		print a2kp1

		z = self.gatingf(a2k, a2kp1)
		print z

		self.a2 = np.dot(z, self.w2)+self.b2
		print self.a2
		return

	'''
		run forward pass online for datapoint x_i for stochastic gradient descent
		NOTE: pass index instead of data
	'''
	def forward_prop_online(self, i):
		# dot() - for 2D arrays equivalent to matrix multiplication!
		if self.epochs == 0:
			x = self.X[i]
			self.a1s = np.dot(self.w1s,x)+self.b1
			print self.a1s
			a11 = np.dot(self.w1s[0],x)
			print a11

			# vectorize
			a2k = self.a1s[0:self.num_as:2]
			print a2k
			a2kp1 = self.a1s[1:self.num_as:2]
			print a2kp1

			z = self.gatingf(a2k, a2kp1)
			print z

			self.a2 = np.dot(z, self.w2)+self.b2
			print self.a2
		else: 
			# just use previously computed a2 value for index i in total error
			self.z = self.z[i]
			self.a2 = self.a2[i]
		return

	'''
		compute residuals for logarithmic error function online_mode --> stochastic gradient descent
	'''
	def log_res(self, label):
		# labels - N x 1
		dlabel = (label+1)/2
		self.r2 = self.sigmoidf(self.a2)-dlabel
		#rks_sigm = np.array([[self.w1s[i][k]*self.sigmoidf(self.a1s[i][k]) if (k%2==0) else self.w1s[i][k]*self.sigmoidf(self.a1s[i][k])*self.sigmoidf(-self.a1s[i][k]) for k in range(0,self.num_as)] for i in range(0, self.feature_dim)])
		rks_sigm = np.array([self.w1s[k]*self.sigmoidf(self.a1s[k]) if (k%2==0) else self.w1s[k]*self.sigmoidf(self.a1s[k])*self.sigmoidf(-self.a1s[k]) for k in range(0,self.num_as)])
		self.r1s = np.dot(self.r2, rks_sigm) 
		print len(self.r1s)
		return

	def back_prop_online(self, i):
		label = self.Y[i]
		self.log_res(label)
		x = X[i]
		# assemble gradient
		gw2 = np.dot(self.r2, self.z)
		gb2 = self.r2
		gw1 = np.dot(self.r1s, array(x,)*self.num_as)
		gb1 = self.r1 

		self.w2 = self.w2-theta[i]*gw2
		self.w1s = self.w1s-theta[i]*gw1
		self.b2 = self.b2-theta[i]*gb2
		self.b1 = self.b1-theta[i]*gb1

		return

	def eval_full_err(self):
		# forward pass for all
		forward_prop_batch()
		self.prev_log_err = self.log_err
		log_err_is = np.log(1+np.exp(-self.Y*self.a2))
		self.log_err = sum(log_err_is)/self.num_points


	def gdescent(self):

		while not self.cvg:
			# pick random data point
			i = random.choice(range(self.feature_dim))
			self.forward_prop_online(i)
			self.back_prop_online(i)
			self.eval_full_err() # includes batch forward prop
			if abs(self.prev_log_err - self.log_err) <= self.cvgc:
				print '*** Convergence after ', self.epochs, ' epochs ***'
				self.cvg = True 
				self.print_status()

			self.epochs+=1
			if self.epochs == self.max_epochs:
				print '*** Reached max. num of epochs', self.max_epochs, ' ***'


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
		Artificial XOR Problem
		four datapoints x in R2, h1 = 4
	'''
	#xormlp = MultiLayerPerceptron(4, 2)
	d = scipy.io.loadmat('../mnist/mp_3-5_data.mat') # corresponding MAT file
	data = d['Xtrain']    # Xtest for test data
	labels = d['Ytrain']  # Ytest for test labels

	print 'Finished loading',data.shape[0],'datapoints'
	print 'With',data.shape[1],'components each'

	'''
		split data
	'''
	rand_perm = np.random.permutation(data.shape[0])
	perm_data = data[rand_perm]
	perm_labels = labels[rand_perm]

	train_len = 2*data.shape[0]/3
	train_data = perm_data[0:train_len]
	train_labels = perm_data[0:train_len]

	h1 = 10
	mlp = MultiLayerPerceptron(h1, train_data, train_labels)
	mlp.gdescent()
	'''	
	feature_dim = 784 # length(x)

	fake_x = np.array([np.random.normal(.5,.25) for i in range(0,784)])

	h1 = 10
	Nact = 2*h1
	# initialize w1q and w2q
	mlp = MultiLayerPerceptron(h1, feature_dim)
	mlp.forward_prop(fake_x)
	mlp.log_res(1)
	'''

	return

if __name__ == "__main__":
	main()		

