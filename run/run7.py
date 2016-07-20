import collections
import generic_run

LABEL = 'Fri3'
param_sets = [[('random_seed', seed),
               ('nmaps', 128),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ]
              for task in
              'rev incr bmul badd'.split()
              for seed in range(10)
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
