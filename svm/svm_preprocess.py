import numpy as np
import scipy.io
import random

def save_data(data, labels, name):
	fn_td = str(name)+'_data.npy'
	fn_lb = str(name)+'_labels.npy'
	np.save(fn_td, data)
	print 'Saved '+str(name)+' data to',fn_td
	np.save(fn_lb, labels)
	print 'Saved '+str(name)+' labels to',fn_lb

	return

def main():
	d = scipy.io.loadmat('../mnist/mp_4-9_data.mat') # corresponding MAT file
	data = d['Xtrain']    # Xtest for test data
	labels = d['Ytrain']  # Ytest for test labels

	print 'Finished loading',data.shape[0],' training datapoints'
	print 'With',data.shape[1],'components each'

	test_data = d['Xtest']
	test_labels = d['Ytest']

	print 'Finished loading',test_data.shape[0],' test datapoints'
	print 'With',test_data.shape[1],'components each'

	'''
		shuffle and then save whole dataset
	'''

	rand_perm = np.random.permutation(data.shape[0])
	train_data = data[rand_perm]
	train_labels = labels[rand_perm]

	print train_data
	print np.where(train_data > 0)
	c_max = np.max(train_data)
	print c_max
	c_min = np.min(train_data)
	print c_min

	train_data = (train_data-c_min*1.0)/(c_max-c_min)
	print train_data 
	print np.where(train_data > 0)
	print train_data[234]

	# WRONG train_data[:,:] = (train_data[:,:]-c_min*1.0)/(c_max-c_min)

	train_len = train_data.shape[0]
	print train_len

	save_data(train_data, train_labels, 'svm_training')

	'''
		save subset of 50 patterns 
	'''

	num_samples = 50
	rand_ind = random.sample(range(train_len), num_samples)	
	train_50 = train_data[rand_ind]
	print np.where(train_50 > 0)
	print train_50[3]
	label_50 = train_labels[rand_ind]

	save_data(train_50, label_50, 'svm_'+str(num_samples)+'_training')

	''' 
		Save test_data
	'''

	test_data = (test_data-c_min*1.0)/(c_max-c_min)
	save_data(test_data, test_labels, 'svm_test')	

	return

if __name__ == "__main__":
	main()