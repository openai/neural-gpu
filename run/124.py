import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import collections
from kube import run_openai_cluster as run
#from kube import run

LABEL = 'July-22-03-curriculum'

param_sets = [[('random_seed', seed),
               ('max_steps', 200000),
               ('forward_max', 201),
               ('nmaps', nm),
               ('progressive_curriculum', True),
               ('task', task),
               rx_step,
               ]
              for seed in range(25)
              for task in ['sbcopy,baddt',
                           'sbcopy,baddt,sbaddt',
                           'baddt,sbcopy,sbaddt',
                           'sbcopy,baddzt',
                           'sbcopy,baddzt,sbaddzt',
                           'baddzt,sbcopy,sbaddzt',
                           'sbcopy,baddet',
                           'sbcopy,baddet,sbaddet',
                           'baddet,sbcopy,sbaddet',
              ]
              for nm in [128]
              for rx_step in [('rx_step', 1), None]
              ]

print "Running", len(param_sets), "jobs"
# Remove Nones 
param_sets = [[p for p in ps if p] for ps in param_sets]

param_sets = map(collections.OrderedDict, param_sets)
run.parser.set_defaults(label=LABEL)
#print len(param_sets)
run.main(param_sets)
