import collections
import generic_run

LABEL = 'June-21-03-drop_nmaps'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ('nmaps', nm),
               ('input_height', 2),
               ]
              for seed in range(8)
              for task in ['badde', 'qadde']
              for nm in [3,6,12,24]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
