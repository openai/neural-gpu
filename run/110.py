import collections
import generic_run

LABEL = 'July-05-02-baselines'

param_sets = [[('random_seed', seed),
               ('nmaps', nm),
               ('task', task),
               ]
              for seed in range(5)
              for task in ['sbadde,sbaddet', 'sbadd,sbaddt', 'scopy', 'sdup',
                           'sbaddet,sbadde', 'sbaddt,sbadd', ]
              for nm in [24, 128]
#              for gs in [0, 6, 24]
              #[0,2,4,5,6,7,8,9]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
