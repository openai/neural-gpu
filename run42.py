import collections
import generic_run

LABEL = 'June-14-03-lastout'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ('do_lastout', v),
               ]
              for seed in range(8)
              for task in ['badd']
              for v in [False, True]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
