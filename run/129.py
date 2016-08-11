import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import collections
from kube import run_openai_cluster as run
#from kube import run

LABEL = 'July-27-11-nmaps-step'

param_sets = [[('random_seed', seed),
               ('max_steps', 40000),
               ('forward_max', 201),
               ('nmaps', nm),
               ('batch_size', bs),
               ('task', task),
               ('progressive_curriculum', 2),
               rx_step,
               ]
              for seed in range(10)
              for task in ['scopy', 'sbcopy',
                           'sbaddt', 'sbaddet', 'baddt', 'baddet', 'baddzt',
                           'bmul', 'sbmul', 'sdup', 'qaddt', 'qmul',
                           'bmule', 'sbmule',
                           'bmul,sbmul', 'baddt,sbaddt',
                           'sbcopy,baddt,sbaddt', 'sbcopy,baddet,sbaddet',
                           'sbcopy,baddet,sbaddet,baddt,sbaddt',
                           'bmul,qmul', 'bmul,sbmul', 'sbaddet,sbaddt',
              ]
              for (nm, bs) in zip([128, 192],
                                  [32, 16])
              for rx_step in [('rx_step', 1), None]
              ]

print "Running", len(param_sets), "jobs"

# Remove Nones 
param_sets = [[p for p in ps if p] for ps in param_sets]

param_sets = map(collections.OrderedDict, param_sets)
run.parser.set_defaults(label=LABEL)
#print len(param_sets)
run.main(param_sets)
