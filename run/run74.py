import collections
import generic_run

LABEL = 'June-23-01-boundedvalues'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('input_height', 2),
               ('task', task),
               ('nmaps', nm),
               ]
              for seed in range(8)
              for task in ['badde,baddet', 'baddet,badde']
              for nm in [6,24]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
