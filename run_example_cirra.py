import collections
import generic_run

LABEL = 'test-example-directory'

param_sets = [[('nmaps', nm),
               ('task', task),
               ('progressive_curriculum', 5),
               ('random_seed', seed),
               ]
              for seed in range(2)
              for task in ['bmul', 'mul', 'bmul,mul', 'bmul,qmul,mul']
              for nm in [24, 128, 256]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
