import collections
import generic_run

LABEL = 'June-16-04-add-attention'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ('num_attention', na),
               ]
              for seed in range(8)
              for task in ['badd', 'qadd', 'badd-qadd']
              for na in [0,3,5,10]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
