import collections
import generic_run

LABEL = 'June-14-07-mixed1'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ('do_lastout', True),
               ('do_attention', v),
               ]
              for seed in range(6)
              for task in ['badd', 'qadd', 'badd-qadd']
              for v in [False, True]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
