import collections
import generic_run

LABEL = 'Sat1'
ON_SERVERS = '14,15,16,17,18'
param_sets = [[('random_seed', seed),
               ('nmaps', nmaps),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', 'rev'),
               ]
              for nmaps in
              [6,12,24,48,96]
              for seed in range(8)
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)
generic_run.parser.set_defaults(on_servers=ON_SERVERS)

generic_run.main(param_sets)
