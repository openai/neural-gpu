import collections
import generic_run

LABEL = 'June-13-3-withcutoffs'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ('cutoff_tanh', 1.2),
               ]
              for seed in range(8)
              for task in ['badd']
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
