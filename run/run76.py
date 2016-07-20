import collections
import generic_run

LABEL = 'June-23-02-binarized2'

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
              for binarization in [1e0, 1e+1,]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
