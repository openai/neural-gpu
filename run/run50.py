import collections
import generic_run

LABEL = 'June-14-07-mul'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ]
              for seed in range(16)
              for task in ['bmul']
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
