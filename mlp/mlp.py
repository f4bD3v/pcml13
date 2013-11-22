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

'''
	sigmoid function
	parameter: vector of activation values
'''
def sigmoidf(x):
	expx = np.exp(x)	
	return 1/(1+expx)

'''
	gating function for pairs of activation values (vectors)
	parameters: a2k (vector), a2kp1 (vector)
'''
def gatingf(a2k, a2kp1):
	return a2k*np.exp(a2kp1)

def forwardpass(Nact, h1, x, w1s, w2):

	# dot() - for 2D arrays equivalent to matrix multiplication!
	a1s = np.dot(w1s,x)
	print a1s
	a11 = np.dot(w1s[0],x)
	print a11

	a2k = a1s[0:len(a1s):2]
	print a2k
	a2kp1 = a1s[1:len(a1s):2]
	print a2kp1

	z = gatingf(a2k, a2kp1)
	print z

	a2 = np.dot(w2, z)
	print a2
	return a2 

# split data into TRAINING set (2/3) and VALIDATION set (1/3); TEST set is provided separately

# PREPROCESSING DATA: normalize input patterns to have coefficients in [0,1]
# get max coefficient and min coefficient

# scale and center features

def main():
	feature_dim = 784 # length(x)

	fake_x = np.array([np.random.normal(.5,.25) for i in range(0,784)])

	h1 = 10
	Nact = 2*h1
	# initialize w1q and w2q

	# w1s Nact x feature_dim 
	w1s = np.array([[np.random.normal(0,1) for i in range(0,feature_dim)] for i in range(0,Nact)])

	#print "w1s: "+str(w1s)

	w2 = np.array([np.random.normal(0,1) for i in range(0,h1)])

	#print "w2: "+str(w2)

	forwardpass(Nact, h1, fake_x, w1s, w2)

	return

if __name__ == "__main__":
	main()		