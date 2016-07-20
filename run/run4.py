import collections
import generic_run


param_sets = [[('random_seed', seed),
               ('cutoff', cutoff),
               ('cutoff_tanh', tanh_cutoff),
               ]
              for cutoff, tanh_cutoff in
              [(0, 0), (1.2, 1.2), (1.2, 0)]
              for seed in range(2)
              ]
param_sets = map(collections.OrderedDict, param_sets)

generic_run.main(param_sets)
