import collections
import generic_run

LABEL = 'June-09-3'
param_sets = [[('random_seed', seed),
               ('nmaps', nmaps),
               ('batch_size', batch_size),
               ('cutoff', 1.2),
               ('cutoff_tanh', 1.2),
               ('max_length', 41),
               ('forward_max', 401),
               ('task', task),
               ]
              for seed in range(8)
              for nmaps in [24]
              for batch_size in [32]
              for task in ['badd']
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
