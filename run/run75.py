import collections
import generic_run

LABEL = 'June-23-02-binarized'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('input_height', 2),
               ('task', task),
               ('nmaps', nm),
               ('do_binarization', binarization),
               ]
              for seed in range(8)
              for task in ['baddet,badde']
              for nm in [6,24]
              for binarization in [1e-1, 1e-2, 1e-3, 1e-4,]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
