import collections
import generic_run

LABEL = 'June-13-12-simplerev'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ]
              for seed in range(8)
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
