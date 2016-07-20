import collections
import generic_run

LABEL = 'June-20-04-aligned'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ('input_height', 2)
               ]
              for seed in range(8)
              for task in ['badd', 'badde']
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
