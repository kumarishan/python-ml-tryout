print __doc__

import datetime
import numpy as np
import pylab as pl
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from sklearn.hmm import GaussianHMM

start_date = datetime.date(1995, 1, 1)
end_date = datetime.date(2012, 1, 1)
quotes = quotes_historical_yahoo("INTC", start_date, end_date)
if len(quotes) == 0:
    raise SystemExit

dates = np.array([q[0] for q in quotes], dtype=int)
close_v = np.array([q[2] for q in quotes])
volume = np.array([q[5] for q in quotes])[1:]

diff = close_v[1:] - close_v[:-1]
dates = dates[1:]
close_v = close_v[1:]

X = np.column_stack([diff, volume])

print "fitting to HMM and decoding..."
n_models = 10

n_components = []
models = []
hidden_states = []

for i in xrange(n_models):
    n_components.insert(i, i+1)
    models.insert(i, GaussianHMM(n_components[i], covariance_type="diag", n_iter=1000))
    models[i].fit([X])
    hidden_states.insert(i, models[i].predict(X))

for i in xrange(n_models):
    print "Transition matrix for %dth model is" % i
    print models[i].transmat_
    print ""
    print "score = ", models[i].score(X)

for j in xrange(n_models):
    print "means and vars of each hidden state of %dth model" % j
    for i in xrange(n_components[j]):
        print "%dth.hidden state" % i
        print "mean = ", models[j].means_[i]
        print "var = ", np.diag(models[j].covars_[i])
        print ""

years = YearLocator()
months = MonthLocator()
yearsFmt = DateFormatter("%Y")
figs = []

for k in xrange(n_models):
    figs.insert(k, pl.figure())
    ax = figs[k].add_subplot(111)

    for i in xrange(n_components[k]):
        idx = (hidden_states[k] == i)
        ax.plot_date(dates[idx], close_v[idx], 'o', label='%dth hidden state ' % i)
    ax.legend()

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.autoscale_view()

    ax.fmt_xdata = DateFormatter('%Y-%m-%d')
    ax.fmt_ydata = lambda x: '$%1.2f' % x
    ax.grid(True)

    figs[i].autofmt_xdate()

pl.show()

