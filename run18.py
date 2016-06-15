import collections
import generic_run

LABEL = 'June-09-2'
ON_SERVERS = '10,14'
param_sets = [[('random_seed', seed),
               ('nmaps', nmaps),
               ('batch_size', batch_size),
               ('cutoff', 0.0),
               ('cutoff_tanh', 0.0),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ]
              for seed in range(8)
              for nmaps in [24, 128]
              for batch_size in [32]
              for task in ['badd']
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)
generic_run.parser.set_defaults(on_servers=ON_SERVERS)

generic_run.main(param_sets)
