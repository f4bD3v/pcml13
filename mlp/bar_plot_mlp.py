import numpy as np
import matplotlib.pyplot as plt

plt.figure('bar_plot')
plt.title('Training and Testing zero-one errors by classifier')
N = 2
ind = np.arange(2)
print ind
mlp1_test = 0.01845
mlp1_std = 0.002734
mlp2_test = 
train = np.array([svm_train, mlp_train])
test = np.array([svm_test, mlp_test])

width = 0.2
plt.bar(ind, train, width, color='blue', label = "training error") 
plt.bar(ind+width, test, width, yerr=[0, 0.003], color='red', label = "testing error")
plt.xlim(-width, len(ind)+width)
plt.xticks(ind+width, np.array(['SVM', 'MLP']), rotation = 45, fontsize=12)
plt.legend(loc='upper right')
plt.show()