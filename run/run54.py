import collections
import generic_run

LABEL = 'June-15-08-qmul'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('nmaps', nmaps),
               ('task', task),
               ]
              for seed in range(8)
              for task in ['qmul']
              for nmaps in [24]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
