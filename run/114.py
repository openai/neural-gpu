import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import collections
from kube import run

LABEL = 'July-07-07-testkube'

param_sets = [[('random_seed', seed),
               ('nmaps', nm),
               ('task', task),
               ('do_outchoice', oc),
               ('progressive_curriculum', ',' in task)
               ]
              for seed in range(2)
              for task in ['scopy', 'sdup']
              for nm in [24]
              for oc in [False]
#              for gs in [0, 6, 24]
              #[0,2,4,5,6,7,8,9]
              ]


param_sets = map(collections.OrderedDict, param_sets)
run.parser.set_defaults(label=LABEL)

run.main(param_sets)
