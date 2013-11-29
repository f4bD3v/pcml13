import numpy as np
import scipy.io
import random

def main():
	d = scipy.io.loadmat('../mnist/mp_4-9_data.mat') # corresponding MAT file
	data = d['Xtrain']    # Xtest for test data
	labels = d['Ytrain']  # Ytest for test labels

	test_data = d['Xtest']
	test_labels = d['Ytest']

	print 'Finished loading',data.shape[0],'datapoints'
	print 'With',data.shape[1],'components each'



	'''
		shuffle and then save whole dataset
	'''

	rand_perm = np.random.permutation(data.shape[0])
	train_data = data[rand_perm]
	train_labels = labels[rand_perm]

	c_max = np.max(train_data)
	c_min = np.min(train_data)

	train_data[:,:] = (train_data[:,:]-c_min*1)/(c_max-c_min)
	train_len = train_data.shape[0]

	fn_td = 'svm_training_data.npy'
	fn_lb = 'svm_training_labels.npy'
	np.save(fn_td, train_data)
	print 'Saved training data to',fn_td
	np.save(fn_lb, train_labels)
	print 'Saved training labels to',fn_lb



	'''
		downsample bitmaps
		- take a subset of vectors with a subset of components
		DON'T DO THIS - waste of time

		save subset of 50 patterns 
	'''

	num_samples = 50
	rand_ind = random.sample(range(train_len), num_samples)	
	train_50 = train_data[rand_ind]
	label_50 = train_labels[rand_ind]

	fn_td = 'svm_'+str(num_samples)+'_training_samples.npy'
	fn_lb = 'svm_'+str(num_samples)+'_training_labels.npy'
	np.save(fn_td, train_50)
	print 'Saved',num_samples,'training data samples to',fn_td
	np.save(fn_lb, label_50)
	print 'Saved according training labels to',fn_lb



	''' 
		Save test_data
	'''

	test_data[:,:] = (test_data[:,:]-c_min*1)/(c_max-c_min)
	fn_td = 'svm_test_data.npy'
	fn_lb = 'svm_test_labels.npy'
	np.save(fn_td, test_data)
	print 'Saved test data to',fn_td
	np.save(fn_td, test_labels)
	print 'Saved test labels to',fn_td

	return

if __name__ == "__main__":
	main()