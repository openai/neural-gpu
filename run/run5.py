import collections
import generic_run

LABEL = 'Fri1'
param_sets = [[('random_seed', seed),
               ('cutoff', cutoff),
               ('cutoff_tanh', tanh_cutoff),
               ('do_attention', do_attention),
               ]
              for cutoff, tanh_cutoff in
              [(0, 0), (1.2, 0), (1.2, 1.2)]
              for do_attention in [False, True]
              for seed in range(4)
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
