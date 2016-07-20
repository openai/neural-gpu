import collections
import generic_run

LABEL = 'June-09-8'
generic_run.parser.set_defaults(dir='neural_gpu_really_old')

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('nmaps', 24),
               ('forward_max', 401),
               ('task', task),
               ]
              for seed in range(8)
              for task in ['badd']
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
