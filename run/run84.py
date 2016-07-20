import collections
import generic_run

LABEL = 'June-24-07-shifter'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('input_height', 2),
               ('task', task),
               ('nmaps', nm),
               ('height', h),
               ('do_binarization', binarization),
               ('do_shifter', True),
               ]
              for seed in range(3)
              for task in ['badde,baddet', 'badd,baddt']
              for nm, h in [(24,4), (12,8), (6,16)]
              for binarization in [1e-1, 1e-2]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
