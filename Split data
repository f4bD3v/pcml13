from numpy import *
import random
import scipy.io
import matplotlib.pyplot as plt

d = scipy.io.loadmat('mp_3-5_data.mat') # corresponding MAT file
data = d['Xtrain']    # Xtest for test data
labels = d['Ytrain']  # Ytest for test labels
s=[]
traindata=[]
trainlabels=[]
testdata =[]
testlabels=[]

print 'Finished loading',data.shape[0],'datapoints'

#plt.imshow(data[3000].reshape(28,28).T)
#plt.hist(labels,[-1,0,1])
 
#pairing the data with the labels
for i in range(6000):
    s.append((d['Xtrain'][i],d['Ytrain'][i]))
   #shuffling the pairs
random.shuffle(s,random.random)
#training set is 2 thirders the original set ( 0 to 4000)
for i in range(4000):
   u , v = s[i]
   traindata.append(u)
   trainlabels.append(v)
# test set
for i in range(2000):
   u , v = s[i+4000]
   testdata.append(u)
   testlabels.append(v)

 
    
