import collections
import generic_run

LABEL = 'Fri2'
param_sets = [[('random_seed', seed),
               ('task', task),
               ]
              for task in
              'rev incr bmul badd mul'.split()
              for do_attention in [False, True]
              for seed in range(8)
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
