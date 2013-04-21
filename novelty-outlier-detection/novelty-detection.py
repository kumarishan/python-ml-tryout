import numpy as np
import pylab as pl
import matplotlib.font_manager 
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]

X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == -1].size

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
print Z
Z = Z.reshape(xx.shape)
print Z

pl.title("Novelty detection")
pl.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=pl.cm.Blues_r)
a = pl.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
pl.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

b1 = pl.scatter(X_train[:, 0], X_train[: , 1], c='white')
b2 = pl.scatter(X_test[:, 0], X_test[:, 1], c='green')
c = pl.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')

pl.axis('tight')
pl.xlim((-5, 5))
pl.ylim((-5, 5))
pl.legend([a.collections[0], b1, b2, c],
	["learned frontier", "training observation", "new regular observation", "new abnormal observation"],
	loc="upper left",
	prop=matplotlib.font_manager.FontProperties(size=11)
	)
pl.xlabel(
	"error train: %d/200 ; error novels regular: %d/20 ;"
	"errors novel abnormal: %d/20"
	% (n_error_train, n_error_test, n_error_outliers))

pl.show()
