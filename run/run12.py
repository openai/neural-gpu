import collections
import generic_run

LABEL = 'Tue1'
ON_SERVERS = '11,12,13,14,16,17,18,19'
param_sets = [[('random_seed', seed),
               ('nmaps', nmaps),
               ('cutoff', 0.0),
               ('cutoff_tanh', 0.0),
               ('binary_activation', d),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ]
              for nmaps in [24]
              for d in [1.0, 0.99, 0.98, 0.95, 0.9, 0.8, 0.5]
              for task in ['rev']
              for seed in range(5)
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)
generic_run.parser.set_defaults(on_servers=ON_SERVERS)

generic_run.main(param_sets)
