import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import collections
from kube import run

LABEL = 'July-14-03-changes'

param_sets = [[('random_seed', seed),
               ('forward_max', 201),
               ('nmaps', nm),
               ('task', task),
               option,
               ]
              for seed in range(10)
              for task in ['scopy', 'sdup', 'bmul'] +
                          [x for y in ['sbadd', 'badd',
                                       'badde', 'sbadde',
                                       'baddz']
                           for x in ['%s,%st' % (y,y),
                                     '%st,%s' % (y,y)]]
              for nm in [24, 128]
              for option in
              [('cutoff', 0.0),
               ('dropout', 0.0),
               ('grad_noise_scale', 0.1),
               ('rx_step', 1),
#               ('nconvs', 3),
#               ('do_resnet', True),
               ('do_binarization', 0.02),
#               ('do_avgout',True),
               ]
              if not (nm == 128 and option[0] == 'do_binarization')
              ]


param_sets = map(collections.OrderedDict, param_sets)
run.parser.set_defaults(label=LABEL)
#print len(param_sets)
run.main(param_sets)
