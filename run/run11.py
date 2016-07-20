import collections
import generic_run

LABEL = 'Mon1'
ON_SERVERS = '11,12,13,14,16,17,18,19'
param_sets = [[('random_seed', seed),
               ('nmaps', nmaps),
               ('cutoff', c),
               ('cutoff_tanh', c),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ]
              for nmaps in [24]
              for c in [0.0, 1.2]
              for task in ['rev', 'badd']
              for seed in range(16)
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)
generic_run.parser.set_defaults(on_servers=ON_SERVERS)

generic_run.main(param_sets)
