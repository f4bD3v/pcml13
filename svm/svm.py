import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold

'''
	SMO with 10-fold cross-validation
	Gaussian Kernel 

	TODO: 
	plot SVM criterion as function of SMO iterations (every 20 steps), 
	plot the value of the convergence criterion, based on KKT condition violations. (vertical axis should have logarithmic scale)

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
	def reset(self, X_train, X_valid, Y_train, Y_valid, tao, C):
		self.tao = tao
		self.C = C
		self.X = X_train
		self.num_patterns = self.X.shape[0]

		self.X_valid = X_valid
		self.Y_valid = Y_valid

		self.t= np.zeros(self.num_patterns, dtype=np.float64)
		self.t = Y_train
		self.alpha = np.zeros((self.num_patterns,1), dtype=np.float64)
		#print self.alpha.shape
		self.stop_coeff = 1E-4
		self.set_K()
		self.f = -self.t
		# Ilow initialized as all indices of points with negative labels
		self.Ilow = np.where(self.t < 0)[0]
		# Iup initialized as all indices of points with negative labels
		self.Iup = np.where(self.t > 0)[0]

		self.curr_pair = (-1,-1)
		self.iter = 0
		self.iters = np.empty(0)
		self.criterions = np.empty(0)
		self.violations = np.empty(0)
		self.zerone_train = np.empty(0)
		self.zerone_valid = np.empty(0)

		return

	'''
		compute kernel matrix
	'''
	def set_K(self):
		onen = np.ones((self.num_patterns,1))
		XXT = np.dot(self.X, self.X.T)
		d = np.diag(XXT)
		A = .5*np.outer(d,onen)+.5*np.outer(onen,d)-XXT
		self.K = np.exp(-self.tao*A)
		#print "K", self.K
		return

	def set_K_class(self, svs):
		XsvT = np.dot(self.X_valid, svs.T)
		#print XsvT
		sq_norm_X = np.diag(np.dot(self.X_valid, self.X_valid.T))
		sq_norm_sv = np.diag(np.dot(svs, svs.T))
		sv_ones = np.ones(svs.shape[0])
		X_ones = np.ones(self.X_valid.shape[0])
		#print np.outer(sq_norm_X, sv_ones).shape
		#print np.outer(sq_norm_sv, X_ones).T.shape
		#print XsvT.shape
		A = .5*np.outer(sq_norm_X, sv_ones)+.5*np.outer(X_ones, sq_norm_sv)-XsvT
		#print A
		self.K_class = np.exp(-self.tao*A)

	'''
		select indices of most validated pair alpha_i, alpha_j
	'''
	def select_mv_pair(self):
		fi_low_max = np.argmax(self.f[self.Ilow])
		#print "fi_low_max", fi_low_max
		i_low = self.Ilow[fi_low_max]
		#print "i_low",i_low
		fi_up_min = np.argmin(self.f[self.Iup])
		#print "fi_up_min", fi_up_min
		i_up = self.Iup[fi_up_min]
		#print "i_up",i_up

		if self.curr_pair == (i_low, i_up):
			i_low_ind = np.where(self.Ilow == i_low)[0]
			print i_low_ind
			print self.Ilow
			self.Ilow = np.delete(self.Ilow, i_low_ind)
			i_up_ind = np.where(self.Ilow == i_up)[0]
			self.Iup = np.delete(self.Iup, i_up_ind)
			fi_low_max = np.argmax(self.f[self.Ilow])
			i_low = self.Ilow[fi_low_max]
			fi_up_min = np.argmin(self.f[self.Iup])
			i_up = self.Iup[fi_up_min]

		self.curr_pair = (i_low, i_up)
		self.bup = self.f[i_up]
		self.blow = self.f[i_low]

		#print self.blow
		#print self.bup
		if self.blow <= self.bup + 2.0*self.stop_coeff:
			i_low = -1
			i_up = -1
			self.curr_pair = (-1,-1)

		return

	def set_L_H(self, i, j):
		sw = self.sigma*self.w
		self.L = np.maximum(0.0, sw - (self.C if self.sigma==1 else 0.0))
		self.H = np.minimum(self.C, sw + (self.C if self.sigma==-1 else 0.0))

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

	def eval_train_err(self):
		# self.final_sv_ind
		sv_ind = np.where((self.alpha > 0) & (self.alpha <= self.C))[0]
		ys = np.dot(self.K[sv_ind, :].T, self.alpha[sv_ind]*self.t[sv_ind])+np.tile(self.b, (self.X.shape[0],1))
		self.train_output = np.sign(ys)
		self.train_score = len(np.where(self.train_output != self.Y_valid)[0])/self.Y_valid[0]

		return

	def eval_class_err(self):
		sv_ind = np.where((self.alpha > 0) & (self.alpha <= self.C))[0]
		svs = self.X[sv_ind, :]
		self.set_K_class(svs)
		ys = np.dot(self.K_class,self.alpha[sv_ind]*self.t[sv_ind])-np.tile(self.b, (self.X_valid.shape[0],1))
		self.class_output = np.sign(ys)
		self.score = len(np.where(self.class_output != self.Y_valid)[0])

		return
		

	def eval_plot(self):
		print self.iters
		print self.criterions
		print self.violations
		plt.figure('Criterion')
		plt.title('SVM criterion')
		plt.plot(self.iters, self.criterions, color="blue", label="SVM criterion")
		plt.xlim(1, self.iters[-1])
		plt.xlabel('iterations')
		plt.ylabel('$\Phi$')
		plt.legend(loc='upper right')
		plt.show()

		plt.figure('Violations')
		plt.title('SVM criterion, abs(bup-blow)')
		plt.plot(self.iters, self.violations, color="red", label="criterion based on violations")
		plt.xlabel('iterations')
		plt.ylabel('$\Phi$')
		plt.yscale('log')
		plt.legend(loc='upper right')
		plt.show()

		return

	def print_status(self):
		print "sigma", self.sigma
		print "w", self.w
		print "L", self.L
		print "H", self.H
		print "bup",self.bup
		print "blow", self.blow
		return

	def seq_min_opt(self):

		while np.dot(self.alpha.T, self.t)==0.0:

			self.select_mv_pair()
			#if self.rec_pair and self.rec_pair == mv_pair:
			#mv_pair = self.select_mv_pair(mv_pair)

			#print "index pair", self.curr_pair
			i = self.curr_pair[0]
			j = self.curr_pair[1]

			if j == -1:
				self.eval_train_err
				break

			self.sigma = self.t[i]*self.t[j]
			#print "sigma",self.sigma
			self.w = self.alpha[i]+self.sigma*self.alpha[j]

			self.set_L_H(i,j)
			#print self.K[i,i]
			eta = self.K[i,i]+self.K[j,j]-2.*self.K[i,j]
			#print "eta",eta

			nalphaj = 0.0
			if eta > 10E-15:
				unc_alphaj = self.alpha[j]+self.t[j]*(self.f[i]-self.f[j])/eta
				#print "unc_alpha",unc_alphaj
				nalphaj = self.clip_unc_alpha(unc_alphaj)
			# second derivative is negative
			else:
				print "hello"
				vi = self.f[i]+self.t[i]-self.alpha[i]*self.t[i]*self.K[i,i]-self.alpha[j]*self.t[j]*self.K[i,j]
				vj = self.f[j]+self.t[j]-self.alpha[i]*self.t[i]*self.K[i,j]-self.alpha[j]*self.t[j]*self.K[j,j]
				phiH = self.eval_phi_H(i,j, vi, vj)
				phiL = self.eval_phi_L(i,j, vi, vj)
				if phiL > phiH:
					nalphaj = self.H
				else:
					nalphaj = self.L


			#self.print_status()

			# New alpha_i

			nalphai = self.alpha[i]+self.sigma*(self.alpha[j]-nalphaj)

			# update f

			fd = self.t[i]*(nalphai-self.alpha[i])*self.K[:,i]+self.t[j]*(nalphaj-self.alpha[j])*self.K[:,j]
			#print "fd",fd
			# reshape f from (num_points,) to (num_points,1)
			fdd = fd.reshape(fd.shape[0],1)
			#print "fdd",fdd
			self.f = self.f + fdd
			#print self.f

			# Update alphas

			self.alpha[i] = nalphai
			self.alpha[j] = nalphaj
			#print "nalphaj", nalphaj
			#print "nalphai", nalphai

			# update sets I_low, I_up
			'''
			sv_ind = np.where((self.alpha >= 0) & (self.alpha <= self.C))[0]
			if sv_ind.size == 0:
				self.b = 0
			else:
				print sv_ind
				#print self.alpha
				ytildei = np.dot(self.K[:,sv_ind].T, self.alpha*self.t)
				print len(sv_ind)
				self.b = np.sum(self.t[sv_ind]-ytildei)/len(sv_ind)
			'''

			self.b = (self.bup+self.blow)/2
			#print "b", self.b

			#print "Iup before",self.Iup
			#print "Ilow before",self.Ilow

			Izero = np.where((self.alpha > 0) & (self.alpha < self.C))[0]

			I_pos = np.where(((self.t == 1) & (self.alpha == 0)) | ((self.t == -1) & (self.alpha==self.C)))[0]
			self.Iup = np.union1d(Izero, I_pos)
			#print "Iup", self.Iup

			I_neg = np.where(((self.t == -1) & (self.alpha == 0)) | ((self.t == 1) & (self.alpha==self.C)))[0]
			self.Ilow = np.union1d(Izero, I_neg)
			#print "Ilow", self.Ilow 

			#print "alphas",self.alpha

			self.iter+=1
			#print self.iter
			#print self.curr_pair

			if self.iter%20 == 0:
				viol_crit = abs(self.bup-self.blow)
				phi = 1./2*np.sum(self.alpha*(self.t*self.f)-self.alpha)
				phi = np.sum(1./2*np.dot(np.outer(self.alpha*self.t, self.alpha*self.t), self.K.T)-self.alpha)
				self.iters = np.append(self.iters, self.iter)
				self.criterions = np.append(self.criterions, phi)
				self.violations = np.append(self.violations, viol_crit)

		return		

	def classify(self):
		sv_ind = np.where((self.alpha > 0) & (self.alpha <= self.C))[0]
		self.eval_train_err()
		self.eval_class_err()
		self.zerone_train = 1.*self.train_score/self.Y_valid.shape[0]
		self.zerone_valid = 1.*self.score/self.Y_valid.shape[0]
		print "zerone train",self.zerone_train
		print "zerone valid",self.zerone_valid
		misclass_error = (self.score*100.)/(self.Y_valid.shape[0])
		print "misclass error",misclass_error
		print "success perc.",(100-misclass_error)
		self.eval_plot()

		return


def CV(svm, training_data, training_labels, taos, Cs, k):

	CVscores = np.zeros((len(taos), len(Cs)))
	for i in range(len(taos)):
		tao = taos[i]
		for j in range(len(Cs)): 
			C = Cs[j]
			CV_err = 0
			score = 0

			klf = KFold(len(training_labels), k, indices=False)

			for train,cross_valid in klf:
				X_train, X_valid, Y_train, Y_valid = training_data[train], training_data[cross_valid], training_labels[train], training_labels[cross_valid]
				print "num X train",len(X_train)
				print "num Y train",len(Y_train)
				print "num X valid",len(X_valid)
				print "num Y valid",len(Y_valid)

				#print len(X_train)
				svm.reset(X_train, X_valid, Y_train, Y_valid, tao, C)
				svm.seq_min_opt()
				svm.classify()
				print "score", svm.score
				score += svm.score
				CV_err += svm.class_err

			CV_err/=(k*1.)
			print "CV valid. error", CV_err
			CVscores[i,j] = score

	np.save('CVscores_'+str(k)+'_fold', CVscores)

	return

def main():

	svm = C_SupportVectorMachine(10, 5E-4)

	'''
		XOR test
	'''

	'''
	xor_train = np.array([[1,0],[0,0],[0,1],[1,1]])
	xor_labels = np.array([[-1],[1],[-1],[1]])

	svm.reset(xor_train, xor_train, xor_labels, xor_labels, 1E-4,100)
	svm.seq_min_opt()
	print svm.classify()
	'''

	''' 
		preprocessing: random permutation + normalization
	'''

	training_data = np.load('svm_training_data.npy')
	training_labels = np.load('svm_training_labels.npy')

	length = 10
	linC = np.linspace(0,length-2, length-1, endpoint = True)
	Cs = np.power(2, linC)
	coeffs = np.array([np.power(2,i) for i in range(1,length-1)])
	taos = coeffs*1E-3
	print len(taos)

	#CV(svm, training_data, training_labels, taos, Cs, 10)
	scores = np.load('CVscores_10_fold.npy')

	plt.figure(1)
	plt.title('10-fold CV scores')
	print scores.shape
	im = plt.imshow(scores, interpolation='none', vmin = 0, vmax = 300, cmap='bone_r')
	yloc = np.array(range(scores.shape[0]))
	xloc = np.array(range(scores.shape[1]))
	plt.xticks(xloc, Cs, rotation=45)
	plt.yticks(yloc, taos)
	plt.xlabel('C')
	plt.ylabel('$\\tau$')
	plt.grid(True, which='minor', linestyle='-')
	plt.colorbar(shrink=.9)
	plt.show()

	print scores
	min_ind_flat = np.argmin(scores)
	min_ind = np.unravel_index(min_ind_flat, scores.shape)
	print min_ind

	'''	
	print len(min_ind)
	if len(min_ind) > 1:
		np.min(min_ind)
	else:
		tao = taos[min_ind[0]]
		print tao
		C = Cs[min_ind[1]]
		print C
		'''
	tau = taos[min_ind[0]]
	print tau
	C = Cs[min_ind[1]]
	print C

	test_data = np.load('svm_test_data.npy')
	test_labels = np.load('svm_test_labels.npy')

	svm.reset(training_data, test_data, training_labels, test_labels, tau, C)
	svm.seq_min_opt()
	svm.classify()

	return

if __name__ == "__main__":
	main()