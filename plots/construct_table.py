#!/usr/bin/python3
from __future__ import print_function
import fileinput

import sys
import numpy as np
import pandas as pd
import argparse
import glob
import scipy.signal
import os
import yaml

import collections

parser = argparse.ArgumentParser(description='Get scores')

parser.add_argument("--metric", type=str, default='score')
parser.add_argument("--dir", type=str, default='/home/ecprice/large/research/neural_gpu/neural_parsed_logs/newer')
parser.add_argument("--curr", type=bool, default=True)
parser.add_argument("tasks", type=str, nargs='*')
args =  parser.parse_args()

def groupby(lst, num):
    ans = []
    for i in range(0, len(lst), num):
        yield lst[i:i+num]

class Run(dict):
    @property
    def metadata(self):
        return self['metadata']

    def options(self):
        cmd = self.metadata['commandline']
        lst = []
        for arg in cmd[1:]:
            a, b = arg.split('=', 1)
            lst.append((a.lstrip('-'), b))
        args = collections.OrderedDict(lst)
        return args

    @property
    def tasks(self):
        return self.options()['task'].split(',')

    @property
    def version(self):
        d = self.options()
        for k in 'train_dir task forward_max random_seed max_steps'.split():
            del d[k]
        mapping = {'progressive_curriculum': 'curr'}
        for k, v in mapping.items():
            d[v] = d[k]
            del d[k]
        return '-'.join('%s=%s' % (a, b) for (a, b) in d.items())

    def get_value(self, metric):
        if '.' in metric:
            metric, key = metric.split('.')
        else:
            key = 'fraction'
        data = self[metric][self.task]
        res = data[key]
        count = len(data['last'])
        if isinstance(res, list):
            res = int(np.median([x or 200000 for x in data[key]]) / 100)
            if res == 2000:
                res = np.inf
        return (res, count)

def value_to_str(val):
    if val == np.inf:
        res = '$\\infty$'
    elif val is None:
        res = '-'
    elif isinstance(val, float):
        res = str(int(val*100)) + r'\%'
    else:
        res = str(val)
    return res

def pair_to_str(pair):
    if pair is None:
        return '-'
    else:
        return '%s (%s)' % (value_to_str(pair[0]), pair[1])

def load_all_data(dirname):
    files = glob.glob(os.path.join(dirname, '*'))
    results = []
    for fname in files:
        with open(fname) as f:
            results.append(Run(yaml.load(f)))
    return results

all_runs = load_all_data(args.dir)

if not args.tasks:
    s = set([run.task for run in all_runs])
    print('Need task name.  Options:', ' '.join(s))
    sys.exit()

def build_table(all_runs, tasks):
    rows = {}
    for run in all_runs:
        if run.task in tasks:
            d = rows.setdefault(run.version, {})
            assert run.task not in d
            d[run.task] = run
    return rows

def texify(s):
    return r'\texttt{%s}' % (s.replace('_', r'\_'))

def table_to_str(rows, tasks, metric):
    ans = []
    ans.append(' & '.join(['Name', 'Mean'] + list(tasks)))
    for version, runs in sorted(rows.items()):
        values = [runs[t].get_value(metric) if t in runs else None for t in tasks]
        row_strs = ([texify(version.split('=',1)[1])] +
                    [value_to_str(np.mean([v[0] for v in values if v is not None]))] +
                    [pair_to_str(value) for value in values])
        ans.append(' & '.join(row_strs))
    interior =  '\\\\\n'.join(ans)
    table = r'''\begin{tabular}{lc%s}
%s
\end{tabular}''' % ('c'*len(tasks), interior)
    return table

def split_table_to_str(table, tasks, metric, maxcol):
    ans = []
    for lst in groupby(tasks, maxcol):
        ans.append(table_to_str(table, lst, metric))
    return '\n\n\\noindent'.join(ans)

def get_document(rows, tasks, metrics, maxcol = 7):
    table = build_table(rows, tasks)
    results = []
    for metric in metrics:
        s = split_table_to_str(table, tasks, metric, maxcol)
        results.append('\\section{%s}\n%s' % (metric, s))
    return (r'''
\documentclass[11pt,letterpaper]{article}
\usepackage[margin=1in]{geometry}
\begin{document}
%s
\end{document}
    ''' % '\n\\newpage\n'.join(results))

print(get_document(all_runs, args.tasks, args.metric.split(','), 7))
