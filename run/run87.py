import collections
import generic_run

LABEL = 'June-26-06-full'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('input_height', 2),
               ('task', task),
               ('nmaps', nm),
               ('do_binarization', binarization),
               ('do_resnet', rn),
               ('do_batchnorm', bn),
               ]
              for seed in range(5)
              for task in ['baddet,badde',]
              for nm in [24]
              for (binarization, rn, bn) in
              [(1e-1, False, 0),
               (0.0, False, 0),
               (0.0, True, 0),
               (0.0, False, 2),
               (0.0, True, 2),
              ]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
