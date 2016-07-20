import collections
import generic_run

LABEL = 'June-24-05-long-binarized'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('input_height', 2),
               ('task', task),
               ('nmaps', nm),
               ('do_binarization', binarization),
               ]
              for seed in range(8)
              for task in ['badde,baddet']
              for nm in [6]
              for binarization in [1e-1]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
