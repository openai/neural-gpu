import collections
import generic_run

LABEL = 'June-26-04-shifter'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('input_height', 2),
               ('task', task),
               ('nmaps', nm),
               ('height', h),
               ('do_shifter', ds),
               ]
              for seed in range(5)
              for task in ['badde,baddet', 'badd,baddt']
              for nm, h in [(24,4)]
              for ds in [1,2]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
