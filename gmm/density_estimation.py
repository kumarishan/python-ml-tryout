import numpy as np 
import pylab as pl 
from sklearn import mixture

n_samples = 500

np.random.seed(0)
C = np.array([[0,, -0,7], [3.5, .7]])
X_train = np.r_[np.dot(np.random.randn(n_samples, 2), C), np.random.randn(n_samples, 2) + np.array([20, 20])]

clf = mixture.GMM(n_components=2, covariance_type='full')
clf.fit(X_train)

x = np.linspace(-20.0, 30.0)
