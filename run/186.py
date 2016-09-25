import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import collections
#from kube import run_openai_cluster as run
import generic_run as run
#from kube import run

LABEL = 'August-56-missing-col'

#run.parser.set_defaults(session='August-32-expr2')

param_sets = [[('random_seed', seed),
               ('max_steps', 60000),
               ('forward_max', 201),
               ('nmaps', nm),
               ('task', task),
               ('time_till_eval', 4),
               ('always_large', True),
               ]
              for seed in range(9, 11)
              for task in ['badd', 'badde', 'baddet', 'baddt', 'baddz', 'baddzt', 'bmul', 'sbadd', 'sbadde', 'sbaddet', 'sbaddt','scopy', 'sdup']
                           #'mul2,mul3,mul4,mul5,mul6,mul7,mul8,mul9,mul10'
              for nm in [256]
              ]

print "Running", len(param_sets), "jobs"
#asdad
# Remove Nones 
param_sets = [[p for p in ps if p] for ps in param_sets]

param_sets = map(collections.OrderedDict, param_sets)
run.parser.set_defaults(label=LABEL)
#print len(param_sets)
run.main(param_sets)
