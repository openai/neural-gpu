import collections
import generic_run

LABEL = 'June-21-02-aligned-resnet'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ('input_height', 2),
               ('do_resnet', True),
               ('do_batchnorm', bn),
               ]
              for seed in range(8)
              for task in ['badde', 'qadde']
              for bn in [0, 1, 2]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
