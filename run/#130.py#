import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import collections
from kube import run_openai_cluster as run
#from kube import run

LABEL = 'July-27-03-sample'

param_sets = [[('random_seed', seed),
               ('max_steps', 40000),
               ('forward_max', 201),
               ('nmaps', nm),
               ('task', task),
               ('batch_size', bs),
               ('progressive_curriculum', 2),
               rx_step,
               ]
              for seed in range(1)
              for task in ['sbmule',]
              for (nm, bs) in zip([128, 192, 256],
                                  [32, 21, 16])
              for rx_step in [('rx_step', 1), ('rx_step', 2), None]
              #if not (nm > 128 and rx_step != ('rx_step', 1))
              ]

print "Running", len(param_sets), "jobs"
# Remove Nones 
param_sets = [[p for p in ps if p] for ps in param_sets]

param_sets = map(collections.OrderedDict, param_sets)
run.parser.set_defaults(label=LABEL)
#print len(param_sets)
run.main(param_sets)
