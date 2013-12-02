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
		self.sigma = self.t[i]*self.t[j]
		self.w = self.alpha[i]+sigma*self.alpha[j]
		sw = sigma*w
		self.L = np.max(0, sw-(self.C if sigma==1 else 0))
		self.H = np.max(self.C, sw+(self.C if sigma==-1 else 0))
		return 

	def clip_unc_alpha(self, unc_alpha):
		if unc_alpha < self.L:
			unc_alpha = self.L	
		elif unc_alpha > self.H:
			unc_alpha = self.H

		return unc_alpha

	def eval_phi_L(self, i, j):
		Li = self.w-self.sigma*self.L
		phi = (K[i,i]*np.power(Li,2)+K[j,j]*np.power(self.L,2))/2+self.sigma*K[i,j]*Li*self.L+self.t[i]*Li*vi+self.t[j]*self.L*vj-Li-self.L
		return phi

	def eval_phi_H(self, i, j):
		Hi = self.w-self.sigma*self.H
		phi = (K[i,i]*np.power(Hi,2)+K[j,j]*np.power(self.H,2))/2+self.sigma*K[i,j]*Hi*self.H+self.t[i]*Hi*vi+self.t[j]*self.H*vj-Hi-self.H
		return phi


	def seq_min_opt(self):
		while np.dot(self.alpha, self.t)==0:
			mv_pair = self.select_mv_pair()
			print mv_pair
			i = mv_pair[0]
			j = mv_pair[1]

			if j == -1:
				break

			sigma = self.t[i]*self.t[j]
			self.set_L_H(i,j)
			eta = self.K[i,i]+self.K[j,j]-2*self.K[i,j]

			nalphaj = 0
			if eta > 10E-15:
				unc_alphaj = alpha[j]+self.t[j]*(self.f[i]-self.f[j])/eta
				nalphaj = self.clip_unc_alpha(unc_alphaj)
			# second derivative is negative
			else:
				phiH = self.eval_phi_H(i,j)
				phiL = self.eval_phi_L(i,j)
				if phiL > phiH:
					nalphaj = self.H
				else:
					nalphaj = self.L

			nalphai = self.alpha[i]+sigma*(alpha[j]-nalphaj)
			# update f
			self.f = self.f + self.t[i]*(nalphai-self.alpha[i])*K[:,i]+self.t[j]*(nalphaj-self.alpha[j])*K[:,j]
			self.alpha[i] = nalphai
			self.alpha[j] = nalphaj

			# update sets I_low, I_up

		return		


def main():

	''' 
		preprocessing: random permutation + normalization
	'''
	training_data = np.load('svm_50_training_samples.npy')
	training_labels = np.load('svm_50_training_labels.npy')

	svm = C_SupportVectorMachine(100, 1E-4)
	kf = KFold(len(training_labels), 10, indices=False)
	#X_train, X_valid, Y_train, Y_valid = cross-validation.train_test_split(training_data, training_labels, test_size=0.2, random_state=0)	
	for train,test in kf:
		X_train, X_valid, y_train, y_valid = training_data[train], training_data[test], training_labels[train], training_labels[test]
		svm.reset(X_train, X_valid, y_train, y_valid)
		svm.seq_min_opt()



	return

if __name__ == "__main__":
	main()