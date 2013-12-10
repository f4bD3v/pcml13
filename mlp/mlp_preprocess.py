import numpy as np
import scipy.io
import random


def normalize():

        return        

def save_data(data, labels, name):
        fn_td = str(name)+'_data.npy'
        fn_lb = str(name)+'_labels.npy'
        np.save(fn_td, data)
        print 'Saved '+str(name)+' data to',fn_td
        np.save(fn_lb, labels)
        print 'Saved '+str(name)+' labels to',fn_lb

        return

def main():
        # for gitiots
        #d = scipy.io.loadmat('C:\Users\Administrator\Desktop\pcml13-master/mnist/mp_3-5_data.mat') # corresponding MAT file
        # for gitpros
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
        valid_data = perm_data[train_len:]

        c_max = 1.*np.max(train_data)
        c_min = 1.*np.min(train_data)
        
        
	train_data = (train_data-c_min*1.0)/(c_max-c_min)
	train_labels = perm_labels[0:train_len]
        save_data(train_data, train_labels, 'training')


	valid_data = (valid_data-c_min*1.0)/(c_max-c_min)
	valid_labels = perm_labels[train_len:]

        save_data(valid_data, valid_labels, 'validation')

        '''
                downsample bitmaps
                - take a subset of vectors with a subset of components
                DON'T DO THIS - waste of time

                save subset of 50 patterns 
        '''
        num_samples = 50
        rand_ind = random.sample(range(train_len), num_samples)        
        train_data_50 = train_data[rand_ind]
        print len(train_data_50)
        train_label_50 = train_labels[rand_ind]

        save_data(train_data_50, train_label_50, '50_training')

        #num_samples = num_samples/3  
        rand_ind = random.sample(range(len(valid_data)), num_samples)        
        valid_data_50 = valid_data[rand_ind]
        valid_label_50 = valid_labels[rand_ind]

        save_data(valid_data_50, valid_label_50, str(num_samples)+'_validation')
        
        return

if __name__ == "__main__":
        main()
