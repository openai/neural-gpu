import collections
import generic_run

LABEL = 'June-24-021-binarizedcutoff'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('input_height', 2),
               ('task', task),
               ('nmaps', nm),
               ('do_binarization', binarization),
               ('cutoff_tanh', 1.2),
               ]
              for seed in range(8)
              for task in ['baddet,badde', 'badde,baddet']
              for nm in [6,24]
              for binarization in [1e-1,]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
