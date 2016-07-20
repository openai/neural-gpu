import collections
import generic_run

LABEL = 'July-06-02-choices'

param_sets = [[('random_seed', seed),
               ('nmaps', nm),
               ('task', task),
               ('do_outchoice', oc),
               ('progressive_curriculum', ',' in task)
               ]
              for seed in range(2)
              for task in ['scopy', 'sdup', 'scopy,sdup', 'sdup,scopy',
                           'sbadde,sbadd']
              for nm in [24, 128]
              for oc in [False, True]
#              for gs in [0, 6, 24]
              #[0,2,4,5,6,7,8,9]
              ]


param_sets = map(collections.OrderedDict, param_sets)
generic_run.parser.set_defaults(label=LABEL)

generic_run.main(param_sets)
