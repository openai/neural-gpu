"""Command to run 50 instances of test_addition.py"""
from __future__ import print_function

import generic_run as run
import collections

LABEL = 'September-02-add3'

param_sets = [[('seed', seed),
               ]
              for seed in range(2, 52)
              ]

param_sets = map(collections.OrderedDict, param_sets)

print("Running", len(list(param_sets)), "jobs")

run.parser.set_defaults(label=LABEL)
run.parser.set_defaults(program='python test_addition.py')
run.parser.set_defaults(use_full=True)
run.main(param_sets)
