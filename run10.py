import collections
import generic_run

LABEL = 'Sun1'
ON_SERVERS = '6,12,13,14,15,16,17,18'
param_sets = [[('random_seed', seed),
               ('nmaps', nmaps),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ]
              for nmaps in [2,4,6,12]
              for task in ['rev', 'badd']
              for seed in range(8)
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)
generic_run.parser.set_defaults(on_servers=ON_SERVERS)

generic_run.main(param_sets)
