import numpy as np
import scipy.io
import random

def main():
	d = scipy.io.loadmat('../mnist/mp_3-5_data.mat') # corresponding MAT file
	data = d['Xtrain']    # Xtest for test data
	labels = d['Ytrain']  # Ytest for test labels

	print 'Finished loading',data.shape[0],'datapoints'
	print 'With',data.shape[1],'components each'

	'''
		shuffle and split data into training and validation set
	'''
	rand_perm = np.random.permutation(data.shape[0])
	perm_data = data[rand_perm]
	perm_labels = labels[rand_perm]

	train_len = 2*data.shape[0]/3
	train_data = perm_data[0:train_len]
	val_data = perm_data[train_len:]

	c_max = np.max(train_data)
	c_min = np.min(train_data)

	train_data[:,:] = (train_data[:,:]-c_min*1)/(c_max-c_min)
	val_data [:,:] = (val_data[:,:]-c_min*1)/(c_max-c_min)
	train_labels = perm_labels[0:train_len]

	fn_td = 'training_data.npy'
	fn_lb = 'training_labels.npy'
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

	fn_td = str(num_samples)+'_training_samples.npy'
	fn_lb = str(num_samples)+'_training_labels.npy'
	np.save(fn_td, train_data)
	print 'Saved',num_samples,'training data samples to',fn_td
	np.save(fn_lb, train_labels)
	print 'Saved according training labels to',fn_lb
	return

if __name__ == "__main__":
	main()