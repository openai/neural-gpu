import glob
import pylab
import numpy as np

#pylab.ion()

data = np.array([int(open(fname).read().strip()) for fname in glob.glob('cachedlogs/September-0*/*/threshold')])
data.sort()


pylab.clf()
pylab.plot(1-np.arange(len(data)) * 1./ len(data), data, marker='o')
pylab.loglog()
pylab.xlabel('Fraction of runs')
pylab.ylabel('# carries before failure')
pylab.title('Log log plot: carries by run percentile')
pylab.savefig('carry_runs_loglog.pdf')

pylab.clf()
pylab.plot(data, marker='o')
pylab.xlabel('Run')
pylab.ylabel('# carries before failure')
pylab.title('5% of runs carry much better, but still not perfectly')
pylab.savefig('carry_runs.pdf')

