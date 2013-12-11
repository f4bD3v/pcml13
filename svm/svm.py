import numpy as np
import scipy.io
from sklearn.cross_validation import KFold

'''
	SMO with 10-fold cross-validation
	Gaussian Kernel 
'''

class C_SupportVectorMachine:

	'''
		initialize C-SVM
	'''
	def __init__(self, C, tao):
		self.C = C
		self.tao = tao 
		return

	'''
		assign sample sets for each cross-validation step
	'''
	def reset(self, X_train, X_valid, Y_train, Y_valid):
		self.X = X_train
		self.X_valid = X_valid
		self.Y_valid = Y_valid
		self.t = Y_train
		self.num_patterns = self.X.shape[0]
		self.alpha = np.zeros(self.num_patterns)
		self.set_K()
		self.f = -self.t
		# Ilow initialized as all indices of points with negative labels
		self.Ilow = np.where(Y_train == -1)[0]
		# Iup initialized as all indices of points with negative labels
		self.Iup = np.where(Y_train == 1)[0]
		self.rec_pair = ()

		return

	'''
		compute kernel matrix
	'''
	def set_K(self):
		onen = np.ones(self.num_patterns)
		XXT = np.dot(self.X, self.X.T)
		d = np.diag(XXT)
		A = .5*np.outer(d,onen)+.5*np.outer(onen,d)-XXT
		self.K = np.exp(-self.tao*A)
		print "K", self.K
		return

	def set_K_class(self, svs):
		svXT = np.dot(svs, self.X_valid.T)
		print svXT
		sq_norm_X = np.diag(np.dot(self.X_valid, self.X_valid.T))
		sq_norm_sv = np.diag(np.dot(svs, svs.T))
		sv_ones = np.ones(svs.shape[0])
		X_ones = np.ones(self.X_valid.shape[0])
		A = .5*np.outer(sv_ones,sq_norm_X)+.5*np.outer(sq_norm_sv, X_ones)-svXT
		print A
		self.K_class = np.exp(-self.tao*A).T
	'''
		select indices of most validated pair alpha_i, alpha_j
	'''
	def select_mv_pair(self):
		fi_low_max = np.argmax(self.f[self.Ilow])
		i_low = self.Ilow[fi_low_max]
		print "i_low",i_low
		fi_up_min = np.argmin(self.f[self.Iup])
		i_up = self.Iup[fi_up_min]
		print "i_up",i_up

		print "f",self.f
		print "filow",self.f[i_low]
		print "fiup",self.f[i_up]
		if self.f[i_low] <= self.f[i_up] + 2.0*self.tao:
			i_low = -1
			i_up = -1

		# indices of most valuated pair
		return (i_low, i_up)

	def set_L_H(self, i, j):
		self.w = self.alpha[i]+self.sigma*self.alpha[j]
		print "w",self.w
		sw = self.sigma*self.w
		print "sw",sw
		print "C or 0",(self.C if self.sigma==1 else 0)
		self.L = np.maximum(0.0, sw - (self.C if self.sigma==1 else 0.0))
		print "L", self.L
		self.H = np.minimum(self.C, sw + (self.C if self.sigma==-1 else 0.0))
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
		print "alpha shape",self.alpha.shape
		print "t shape",self.t
		while np.dot(self.alpha, self.t)==0.0:

			mv_pair = self.select_mv_pair()
			if self.rec_pair and self.rec_pair == mv_pair:
				break
				mv_pair = self.select_mv_pair()

			self.rec_pair = mv_pair	

			print "index pair", mv_pair
			i = mv_pair[0]
			j = mv_pair[1]

			if j == -1:
				break

			self.sigma = self.t[i]*self.t[j]
			print "sigma",self.sigma

			self.set_L_H(i,j)
			print self.K[i,i]
			eta = self.K[i,i]+self.K[j,j]-2.0*self.K[i,j]
			print eta

			nalphaj = 0.0
			if eta > 10E-15:
				unc_alphaj = self.alpha[j]+self.t[j]*(self.f[i]-self.f[j])/eta
				print "unc_alpha",unc_alphaj
				nalphaj = self.clip_unc_alpha(unc_alphaj)
				print "nalphaj",nalphaj
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

			print nalphaj

			# New alpha_i

			nalphai = self.alpha[i]+self.sigma*(self.alpha[j]-nalphaj)


			# update f

			fd = self.t[i]*(nalphai-self.alpha[i])*self.K[:,i]+self.t[j]*(nalphaj-self.alpha[j])*self.K[:,j]
			# reshape f from (num_points,) to (num_points,1)
			fdd = fd.reshape(fd.shape[0],1)
			self.f = self.f + fdd

			# Update alphas

			self.alpha[i] = nalphai
			self.alpha[j] = nalphaj
			print "nalphaj", nalphaj
			print "nalphai", nalphai

			# update sets I_low, I_up
			sv_ind = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
			if sv_ind.size == 0:
				self.b = 0
			else:
				print sv_ind
				print self.alpha
				ytildei = np.dot(self.K[:,sv_ind].T, self.alpha*self.t)
				print len(sv_ind)
				self.b = np.sum(self.t[sv_ind]-ytildei)/len(sv_ind)

			print "Iup before",self.Iup
			print "Ilow before",self.Ilow
			#print "b",self.b
			#print "f",self.f
			self.Iup = np.where(self.f <= self.b)[0]
			#print "Iup", self.Iup
			self.Ilow = np.where(self.f >= self.b)[0]
			#print "Ilow", self.Ilow 

			print "alphas",self.alpha
		return		

	def classify(self):
		sv_ind = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
		svs = self.X[sv_ind]
		self.set_K_class(svs)
		ys = np.dot(self.K_class,self.alpha[sv_ind]*self.t[sv_ind])+np.tile(self.b, svs.shape[0])

		return ys


def main():

	svm = C_SupportVectorMachine(.1, 1E-4)

	'''
		XOR test
	'''

	'''
	xor_train = np.array([[1,0],[0,0],[0,1],[1,1]])
	xor_labels = np.array([[1],[-1],[1],[-1]])
	print "labels", xor_labels.shape

	svm.reset(xor_train, xor_train, xor_labels, xor_labels)
	svm.seq_min_opt()
	print svm.classify()

	'''
	''' 
		preprocessing: random permutation + normalization
	'''

	training_data = np.load('svm_50_training_data.npy')
	print training_data[0]
	training_labels = np.load('svm_50_training_labels.npy')

	training_data[0]

	klf = KFold(len(training_labels), 10, indices=False)
	#X_train, X_valid, Y_train, Y_valid = cross-validation.train_test_split(training_data, training_labels, test_size=0.2, random_state=0)	
	for train,cross_valid in klf:
		X_train, X_valid, y_train, y_valid = training_data[train], training_data[cross_valid], training_labels[train], training_labels[cross_valid]

		print len(X_train)
		svm.reset(X_train, X_valid, y_train, y_valid)
		print y_train
		#svm.seq_min_opt()
		#print svm.classify()
		break

	test_data = np.load('svm_test_data.npy')

	test_labels = np.load('svm_test_labels.npy')

	return

if __name__ == "__main__":
	main()