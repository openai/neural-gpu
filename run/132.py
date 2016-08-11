import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import collections
from kube import run_openai_cluster as run
#from kube import run

LABEL = 'July-28-01-mulstuff'

param_sets = [[('random_seed', seed),
               ('max_steps', 40000),
               ('forward_max', 201),
               ('nmaps', 128),
               ('task', task),
               ('progressive_curriculum', 2),
               opt,
               opt2
               ]
              for seed in range(9)
              for task in [
                      'bmul,qmul,mul',
                      'bmul,mul',
                      'qmul',
                      'bmul',
                      
                      'sbcopy', 'scopy',
#                           ]+0*[
                           'sbaddt', 'sbaddet', 'baddt', 'baddet', 'baddzt',
                           'bmul', 'sbmul', 'sdup', 'qaddt', 'qmul',
                           'bmule', 'sbmule',
                           'bmul,sbmul', 'baddt,sbaddt',
                           'sbcopy,baddt,sbaddt', 'sbcopy,baddet,sbaddet',
                           'sbcopy,baddet,sbaddet,baddt,sbaddt',
                           'bmul,qmul', 'bmul,sbmul', 'sbaddet,sbaddt',
              ]
              for opt in [('dropout', 0.3), ('dropout', 0.5), None]
              for opt2 in [('output_layer', 3), None]
              ]

print "Running", len(param_sets), "jobs"
asdasd
# Remove Nones 
param_sets = [[p for p in ps if p] for ps in param_sets]

param_sets = map(collections.OrderedDict, param_sets)
run.parser.set_defaults(label=LABEL)
#print len(param_sets)
run.main(param_sets)
