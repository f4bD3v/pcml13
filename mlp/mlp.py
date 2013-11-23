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
import scipy as sp 

class MultiLayerPerceptron:

	def __init__(self, h1, feature_dim, X, Y):
		self.feature_dim = feature_dim
		self.h1 = h1
		self.num_as = 2*h1
		# initialize w's
		# w1s Nact x feature_dim 
		self.w1s = np.array([[np.random.normal(0,(1/self.feature_dim)) for i in range(0,self.feature_dim)] for i in range(0,self.num_as)])
		#print "w1s: "+str(w1s)
		self.w2 = np.array([np.random.normal(0,1/self.h1) for i in range(0,self.h1)])
		#print "w2: "+str(w2)
		# as many bias terms as as outputs in hidden layer
		self.b1 = np.array([np.random.normal(.5,.25) for i in range(0,self.num_as)])
		self.b2 = np.random.normal(.5,.25)
		self.X = X
		self.Y = Y

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
	def forward_prop_batch(self, X):
		# dot() - for 2D arrays equivalent to matrix multiplication!
		self.a1s = np.dot(self.w1s,X.T)+self.b1
		print self.a1s
		a11 = np.dot(self.w1s[0],x[0])
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
		x = X[i]
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
		return

	'''
		compute residuals for logarithmic error function
		online_mode --> stochastic gradient descent
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

	def back_prop(self, i):
		label = Y[i]
		self.log_res(label)
		x = X[i]
		# assemble gradient
		gw2 = np.dot(self.r2, self.z)
		gb2 = self.r2
		gw1 = np.dot(self.r1s, array(x,)*self.num_as)
		gb1 = self.r1 

		self.w2 = -theta[i]*gw2
		self.w1s = -theta[i]*gw1
		self.b2 = -theta[i]*gb2
		self.b1 = -theta[i]*gb1

		return

	def eval_full_err():

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
		four datapoints x€R^2, h1 = 4
	'''

	xormlp = MultiLayerPerceptron(h1=4, 2)

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

