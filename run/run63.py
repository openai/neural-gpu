import collections
import generic_run

LABEL = 'June-16-08-batchnorm_and_resnet'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ('do_batchnorm', bn),
               ('do_resnet', rn),
               ]
              for seed in range(8)
              for task in ['badd']
              for bn in [False, True]
              for rn in [True]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
