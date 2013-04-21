from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE

digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

svc = SVC(kernel='linear', C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
print rfe.ranking_.reshape(digits.images[0].shape)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

import pylab as pl 
pl.matshow(ranking)
pl.colorbar()
pl.title("Ranking of pixels with RFE")
pl.show()