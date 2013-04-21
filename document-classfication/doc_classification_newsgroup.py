import logging
import numpy as np 
from optparse import OptionParser
import sys
from time import time
import pylab as pl 

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

op = OptionParser()
op.add_option("--report", action="store_true", dest="print_report",
			  help='Print detailed classification report.')

op.add_option("--chi2_select",
			  action="store",type="int", dest="select_chi2",
			  help="Select some number of features using chi squared test")

op.add_option("--confusion_matrix", 
			  action="store_true", dest="print_cm",
			  help="Print the confusion matrix")
op.add_option("--top10",
			  action="store_true", dest="print_top10",
			  help="Print ten most descriminative terms per class")

op.add_option("--all_categories",
			  action="store_true", dest="all_categories",
			  help="Whether to use all categories or not")

op.add_option("--use_hashing",
			  action="store_true",
			  help="Use a hashing vectorizer")

op.add_option("--n_features",
			  action="store", type=int, default=2 ** 16,
			  help="n_features when using the hashing vectorizer")

(opts, args) = op.parse_args()
if len(args) > 0:
	op.error("this script takes no arguments.")
	sys.exit()

print __doc__
op.print_help()
print


if opts.all_categories:
	categories = None
else:
	categories = [
		'alt.atheism',
		'talk.religion.misc',
		'comp.graphics',
		'sci.space',
	]

print "Loading 20 newsgroups dataset for categories:"
print categories if categories else "all"

data_train = fetch_20newsgroups(subset='train', categories=categories,
								shuffle=True, random_state=42)

data_test = fetch_20newsgroups(subset='test', categories=categories,
							   shuffle=True, random_state=42)

print 'data loaded'

categories = data_train.target_names

def size_mb(docs):
	return sum(len(s.encode('utf-8')) for s in docs)/ 1e6

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print ("%d documents - %0.3fMB (training set) " % (len(data_train.data), data_train_size_mb))
print ("%d documents - %0.3fMB (test set) " % (len(data_test.data), data_test_size_mb))

print "%d categories" % len(categories)
print

y_train, y_test = data_train.target, data_test.target

print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
if opts.use_hashing:
	vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
								   n_features=opts.n_features)
	X_train = vectorizer.transform(data_train.data)
else:
	vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_df=0.5)
	X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print ("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb/duration))
print "n_samples: %d, n_features: %d" % X_train.shape
print

print "Extracting features from the test dataset using same vectorizer"
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print ("done in %fs at %0.3fMB/s" %(duration, data_train_size_mb/duration))
print "n_samples: %d, n_features: %d" % X_train.shape
print

if opts.select_chi2:
	print ("Extracting %d best features by a chi-squared test" % opts.select_chi2)
	t0 = time()
	ch2 = SelectKBest(chi2, k=opts.select_chi2)
	X_train = ch2.fit_transform(X_train, y_train)
	X_test = 	ch2.transform(X_test)
	print "done in %fs" % (time() - t0)
	print

def trim(s):
	"""Trim strind to fit on terminal (assuming 80-column display)"""
	return s if len(s) <= 80 else s[:77] + "..."

if opts.use_hashing:
	feature_names = None
else:
	feature_names = np.asarray(vectorizer.get_feature_names)

def benchmark(clf):
	print 80 * '-'
	print "Training: "
	print clf
	t0 = time()
	clf.fit(X_train, y_train)
	train_time = time() - t0
	print "train time: %0.3f" % train_time

	t0 = time()
	pred = clf.predict(X_test)
	test_time = time() - t0
	print "test time %0.3f" % test_time

	score = metrics.f1_score(y_test, pred)
	print "f1 score: %0.3f" % score

	if hasattr(clf, 'coef_'):
		print "dimensionality: %d" % clf.coef_.shape[1]
		print "density: %f" % density(clf.coef_)

		if opts.print_top10 and feature_names is not None:
			print "top 10 keywords per class:"
			for i, category in 	enumerate(categories):
				top10 = np.argsort(clf.coef_[i])[-10:]
				print trim("%s: %s" % (
					category, " ".join(feature_names[top10])))
		print

	if opts.print_report:
		print "classification report:"
		print metrics.classification_report(y_test, pred, target_names=categories)

	if opts.print_cm:
		print "confusion matrix:"
		print metrics.confusion_matrix(y_test, pred)

	print
	clf_descr = str(clf).split('(')[0]
	return clf_descr, score, train_time, test_time

results = []

for clf, name in (
		(RidgeClassifier(tol=1e-2, solver="lsqr"),  "RidgeClassifier"),
		(Perceptron(n_iter=50), "Perceptron"),
		(PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
		(KNeighborsClassifier(n_neighbors=10), "kNN")
	):
	print 80 * '='
	print name
	results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
	print 80 * '='
	print "%s penalty" % penalty.upper()
	results.append(benchmark(LinearSVC(loss='l2', penalty=penalty, dual=False, tol=1e-3)))

	results.append(benchmark(SGDClassifier(alpha=0.0001, n_iter=50, penalty=penalty)))


print 80 * '='
print 'Elastic Net penalty'
results.append(benchmark(SGDClassifier(alpha=0.0001, n_iter=50, penalty='elasticnet')))

print 80 * '='
print "Naive Bayes"
results.append(benchmark(BernoulliNB(alpha=0.01)))
results.append(benchmark(MultinomialNB(alpha=0.01)))

class L1LinearSVC(LinearSVC):
	def fit(self, X, y):
		self.transformer_ = LinearSVC(penalty="l1", dual=False, tol=1e-3)
		X = self.transformer_.fit_transform(X, y)
		return LinearSVC.fit(self, X, y)

	def predict(self, X):
		X = self.transformer_.transform(X)
		return LinearSVC.predict(self, X)

print 80 * '='
print 'LinearSVC with L1-based feature selection'
results.append(benchmark(L1LinearSVC()))

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in xrange(4)]

clf_names, score, training_time, test_time = results
training_time =	np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

pl.title("Score")
pl.barh(indices, score, .2, label='score', color='r')
pl.barh(indices + .3, training_time, .2, label='training time', color='g')
pl.barh(indices + .6, test_time, .2, label='test time', color='b')
pl.yticks(())
pl.legend(loc='best')
pl.subplots_adjust(left=.25)

for i, c in zip(indices, clf_names):
	pl.text(-.3, i, c)

pl.show()