import numpy as np
import matplotlib.pyplot as plt

plt.figure('bar_plot')
plt.title('Training and Testing zero-one errors by classifier')
N = 2
ind = np.arange(2)
print ind
svm_train = 0.00050226
mlp_train = 0.00960
train = np.array([svm_train, mlp_train])
svm_test = 0.00904068307383
mlp_test =0.02395
test = np.array([svm_test, mlp_test])

width = 0.2
plt.bar(ind, train, width, yerr=[0,0.00378], color='blue', label = "training error", error_kw=dict(elinewidth=2,ecolor='black')) 
plt.bar(ind+width, test, width, yerr=[0,0.00140], color='red', label = "testing error", error_kw=dict(elinewidth=2,ecolor='black'))

plt.xlim(-width, len(ind)+width)
plt.xticks(ind+width, np.array(['SVM', 'MLP']), rotation = 45, fontsize=12)
plt.legend(loc='upper right')
plt.show()