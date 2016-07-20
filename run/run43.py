import collections
import generic_run

LABEL = 'June-14-04-baddnmaps'

param_sets = [[('random_seed', seed),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ('nmaps', nmaps)
               ]
              for seed in range(8)
              for task in ['badd']
              for nmaps in [24, 128]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
