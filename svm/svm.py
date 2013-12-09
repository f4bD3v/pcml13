import numpy as np
import scipy.io
from sklearn.cross_validation import KFold

'''
	SMO with 10-fold cross-validation
	Gaussian Kernel 
'''

class C_SupportVectorMachine:

	def __init__(self, C, tao):
		self.C = C
		self.tao = tao 
		return

	def reset(self, X_train, X_valid, Y_train, Y_valid):
		self.X = X_train
		self.t = Y_train
		self.num_patterns = self.X.shape[0]
		self.alpha = np.zeros(self.num_patterns)
		self.set_K()
		self.f = -Y_train
		print self.f
		# Ilow initialized as all indices of points with negative labels
		self.Ilow = np.where(Y_train == -1)[0]
		print self.Ilow
		self.Iup = np.where(Y_train == 1)[0]
		print self.Iup

		return

	def set_K(self):
		onen = np.ones((self.num_patterns, self.num_patterns))
		XXT = np.dot(self.X, self.X.T)
		d = np.diag(XXT)
		A = (np.dot(d,onen)+np.dot(onen,d.T))/2-XXT
		self.K = np.exp(-self.tao*A)
		print self.K
		return

	'''
		select indices of most validated pair alpha_i,alpha_j
	'''
	def select_mv_pair(self):
		fi_low_min = np.argmin(self.f[self.Ilow])
		i_low = self.Ilow[fi_low_min]
		fi_up_min = np.argmin(self.f[self.Iup])
		i_up = self.Iup[fi_up_min]

		if self.f[i_low] <= self.f[i_up] + 2*self.tao:
			i_low = -1
			i_up = -1

		# indices most valuated pair
		return (i_low, i_up)

	def set_L_H(self, i, j):
		self.w = self.alpha[i]+self.sigma*self.alpha[j]
		print "w",self.w
		sw = self.sigma*self.w
		print "sw",sw
		self.L = np.maximum(0, sw-(self.C if self.sigma==1 else 0))
		print "L", self.L
		self.H = np.minimum(self.C, sw+(self.C if self.sigma==-1 else 0))
		print "H", self.H

		return 

	def clip_unc_alpha(self, unc_alpha):
		if unc_alpha < self.L:
			unc_alpha = self.L	
		elif unc_alpha > self.H:
			unc_alpha = self.H

		return unc_alpha

	def eval_phi_L(self, i, j, vi, vj):
		Li = self.w-self.sigma*self.L
		phi = (self.K[i,i]*np.power(Li,2)+self.K[j,j]*np.power(self.L,2))/2+self.sigma*self.K[i,j]*Li*self.L+self.t[i]*Li*vi+self.t[j]*self.L*vj-Li-self.L
		return phi

	def eval_phi_H(self, i, j, vi, vj):
		Hi = self.w-self.sigma*self.H
		phi = (self.K[i,i]*np.power(Hi,2)+self.K[j,j]*np.power(self.H,2))/2+self.sigma*self.K[i,j]*Hi*self.H+self.t[i]*Hi*vi+self.t[j]*self.H*vj-Hi-self.H
		return phi


	def seq_min_opt(self):
		while np.dot(self.alpha, self.t)==0:
			mv_pair = self.select_mv_pair()
			print "index pair", mv_pair
			i = mv_pair[0]
			j = mv_pair[1]

			if j == -1:
				break

			self.sigma = self.t[i]*self.t[j]
			print "sigma",self.sigma

			self.set_L_H(i,j)
			eta = self.K[i,i]+self.K[j,j]-2*self.K[i,j]

			nalphaj = 0
			if eta > 10E-15:
				unc_alphaj = self.alpha[j]+self.t[j]*(self.f[i]-self.f[j])/eta
				nalphaj = self.clip_unc_alpha(unc_alphaj)
			# second derivative is negative
			else:
				vi = self.f[i]+self.t[i]-self.alpha[i]*self.t[i]*self.K[i,i]-self.alpha[j]*self.t[j]*self.K[i,j]
				vj = self.f[j]+self.t[j]-self.alpha[i]*self.t[i]*self.K[i,j]-self.alpha[j]*self.t[j]*self.K[j,j]
				phiH = self.eval_phi_H(i,j, vi, vj)
				phiL = self.eval_phi_L(i,j, vi, vj)
				if phiL > phiH:
					nalphaj = self.H
				else:
					nalphaj = self.L

			nalphai = self.alpha[i]+self.sigma*(self.alpha[j]-nalphaj)
			#nalphai = self.w-self.sigma*nalphaj

			# update f
			self.f = self.f + self.t[i]*(nalphai-self.alpha[i])*self.K[:,i]+self.t[j]*(nalphaj-self.alpha[j])*self.K[:,j]
			self.alpha[i] = nalphai
			self.alpha[j] = nalphaj

			# update sets I_low, I_up
			sv_ind = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
			print sv_ind
			print self.alpha
			ytildei = np.dot(self.K[:,sv_ind].T, self.alpha*self.t)
			print len(sv_ind)
			self.b = np.sum(self.t[sv_ind]-ytildei)/len(sv_ind)
			print self.b

			self.Iup = np.where(self.f <= self.b)[0]
			print "Iup", self.Iup
			self.Ilow = np.where(self.f >= self.b)[0]
			print "Ilow", self.Ilow

		return		


def main():

	''' 
		preprocessing: random permutation + normalization
	'''
	training_data = np.load('svm_50_training_samples.npy')
	training_labels = np.load('svm_50_training_labels.npy')

	svm = C_SupportVectorMachine(100, 1E-4)
	klf = KFold(len(training_labels), 10, indices=False)
	#X_train, X_valid, Y_train, Y_valid = cross-validation.train_test_split(training_data, training_labels, test_size=0.2, random_state=0)	
	for train,test in klf:
		X_train, X_valid, y_train, y_valid = training_data[train], training_data[test], training_labels[train], training_labels[test]
		svm.reset(X_train, X_valid, y_train, y_valid)
		svm.seq_min_opt()



	return

if __name__ == "__main__":
	main()