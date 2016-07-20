import collections
import generic_run

LABEL = 'June-27-01-edged'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('input_height', 2),
               ('nmaps', nm),
               ('task', task),
               ('do_binarization', binarization),
               ]
              for seed in range(5)
              for task in ['badde,baddet','baddet,badde']
              for nm in [24]
              for binarization in
              [1e-1, 1e-3, 0.0]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
