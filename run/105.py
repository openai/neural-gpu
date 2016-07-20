import collections
import generic_run

LABEL = 'June-30-05-globalsum'

param_sets = [[('random_seed', seed),
               ('nmaps', nm),
               ('task', task),
#               ('do_binarization', binarization),
               ('do_shifter', ds),
               ('do_globalsum', gs),
               ]
              for seed in range(7)
              for task in ['baddet,badde']
              for nm in [24]
#              for binarization in [1e-1, 0.0]
              for gs in [0, 6, 24]
              for ds in [0]#,5,7]
              #[0,2,4,5,6,7,8,9]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
