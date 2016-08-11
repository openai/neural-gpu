import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import collections
from kube import run_openai_cluster as run
#from kube import run

LABEL = 'July-26-05-binarization'

param_sets = [[('random_seed', seed),
               ('max_steps', 200000),
               ('forward_max', 201),
               ('nmaps', nm),
               ('task', task),
               ('do_binarization', db),
               rx_step,
               ]
              for seed in range(15)
              for task in ['baddet,badde', 'sbaddet,sbadde']
              for nm in [24, 96]
              for rx_step in [('rx_step', 1), None]
              for db in [0, 0.1]
              ]

print "Running", len(param_sets), "jobs"
# Remove Nones 
param_sets = [[p for p in ps if p] for ps in param_sets]

param_sets = map(collections.OrderedDict, param_sets)
run.parser.set_defaults(label=LABEL)
#print len(param_sets)
run.main(param_sets)
