import collections
import generic_run

LABEL = 'June-06-2'
ON_SERVERS = '11,12,13,14,16,19,21,23'
param_sets = [[('random_seed', seed),
               ('nmaps', nmaps),
               ('max_length', 41),
               ('forward_max', 401),
               ('do_attention', attend),
               ('task', task),
               ]
              for seed in range(8)
              for nmaps in [24]
              for task in ['rev', 'badd', 'bmul', 'mix']
              for attend in [False, True]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)
generic_run.parser.set_defaults(on_servers=ON_SERVERS)

generic_run.main(param_sets)
