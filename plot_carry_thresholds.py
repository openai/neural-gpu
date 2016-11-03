import glob
import pylab
import numpy as np

#pylab.ion()

data = np.array([int(open(fname).read().strip()) for fname in glob.glob('cachedlogs/September-0*/*/threshold2')])
data.sort()


pylab.clf()
pylab.plot(1-np.arange(len(data)) * 1./ len(data), data, marker='o')
pylab.loglog()
pylab.xlabel('Fraction of runs')
pylab.ylabel('Carry length with 50% failure')
pylab.title('Carries by run percentile (log log plot)')
pylab.savefig('carry_runs_loglog.pdf')

pylab.clf()
pylab.plot(data, marker='o')
pylab.xlabel('Run')
pylab.ylabel('# carries before failure')
pylab.title('5% of runs carry much better, but still not perfectly')
pylab.savefig('carry_runs.pdf')

