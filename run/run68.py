import collections
import generic_run

LABEL = 'June-20-05-aligned'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ('input_height', 2),
               ('cutoff_tanh', tc),
               ]
              for seed in range(8)
              for task in ['badde', 'qadde']
              for tc in [0.0, 1.2]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
