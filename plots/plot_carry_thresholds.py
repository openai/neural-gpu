import glob
import pylab
import numpy as np
from matplotlib import rc

rc('font',  size='9')
rc('axes', labelsize='large')
rc('lines', linewidth=3)

#pylab.ion()

data = np.array([int(open(fname).read().strip()) for fname in glob.glob('cachedlogs/September-0*/*/threshold2')])
data.sort()

pylab.figure(figsize=(4,4))

pylab.clf()
pylab.plot(1-np.arange(len(data)) * 1./ len(data), data, marker='o')
pylab.loglog()
pylab.xlabel('Fraction of training runs')
pylab.ylabel('Decimal addition carry length with 50% failure')
pylab.title('A small fraction of trials can carry\nover longer intervals (log log plot)')
pylab.tight_layout()
pylab.savefig('../neuralgpu_paper/carry_runs_loglog.pdf')

pylab.clf()
pylab.plot(data[::-1], marker='o')
pylab.xlabel('Run')
pylab.ylabel('# carries before failure')
pylab.title('5% of runs carry much better, but still not perfectly')
pylab.savefig('carry_runs.pdf')

