import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import collections

# Pick one:
#from kube import run
from kube import run_openai_cluster as run

LABEL = 'jonas-testkube'

param_sets = [[('random_seed', seed),
               ('nmaps', nm),
               ('task', task),
               ('do_outchoice', oc),
               ('progressive_curriculum', ',' in task)
               ]
              for seed in range(2)
              for task in ['scopy']
              for nm in [24000]
              for oc in [False]
#              for gs in [0, 6, 24]
              #[0,2,4,5,6,7,8,9]
              ]


param_sets = map(collections.OrderedDict, param_sets)
run.parser.set_defaults(label=LABEL)

run.main(param_sets)
