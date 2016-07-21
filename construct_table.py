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
parser.add_argument("--dir", type=str, default='/home/ecprice/neural_parsed_logs')
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
    def task(self):
        return self.options()['task'].split(',')[0]

    @property
    def version(self):
        d = self.options()
        for k in 'train_dir task forward_max random_seed'.split():
            del d[k]
        return '-'.join('%s=%s' % (a, b) for (a, b) in d.items())

    def get_str(self, metric):
        if '.' in metric:
            metric, key = metric.split('.')
        else:
            key = 'fraction'
        data = self[metric][self.task]
        res = data[key]
        if isinstance(res, list):
            res = np.median(data[key])
        return '%0.2f (%d)' % (data['fraction'], len(data['first-time']))

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
    ans.append(' & '.join(['Name'] + list(tasks)))
    for version, runs in sorted(rows.items()):
        row_strs = ([texify(version.split('=',1)[1])] +
                    [runs[t].get_str(metric) if t in runs else ''
                     for t in tasks])
        ans.append(' & '.join(row_strs))
    interior =  '\\\\\n'.join(ans)
    table = r'''\begin{tabular}{l%s}
%s
\end{tabular}''' % ('c'*len(tasks), interior)
    return table

def split_table_to_str(table, tasks, metric, maxcol):
    ans = []
    for lst in groupby(tasks, maxcol):
        ans.append(table_to_str(table, lst, metric))
    return '\n'.join(ans)

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
