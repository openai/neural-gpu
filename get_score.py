#!/usr/bin/python
import fileinput

import sys
import numpy as np
import pandas as pd
import argparse
import glob
import scipy.signal
import os

parser = argparse.ArgumentParser(description='Get scores')

parser.add_argument("--key", type=str, default=None)
parser.add_argument("--task", type=str, default='plot')
parser.add_argument("--title", type=str, default='')
parser.add_argument("--median", action='store_true')
parser.add_argument("--smoothing", type=int, default='1')
parser.add_argument('files', type=str, nargs='+',
                    help='Log files to examine')

def get_simple_scores(fname, taskname=None):
    with open(fname) as f:
        for line in f:
            words = line.split()
            loc, val = words[:2]
            if taskname and len(words) > 2 and taskname != words[2]:
                continue
            try:
                yield (int(loc), float(val))
            except ValueError:
                break

def get_scores_for_key(fname, key, index=0):
    with open(fname) as f:
        for line in f:
            if line.startswith('step '):
                entries = line.split()
                d = dict(zip(entries[::2], entries[1::2]))
                try:
                    yield (int(d['step']), float(d[key].split('/')[index]))
                except ValueError:
                    break

def create_results(dirname):
    log0_fname = dirname + '/log0'
    if os.path.exists(log0_fname):
        with open(dirname+'/results', 'w') as f:
            with open(log0_fname) as f2:
                for line in f2:
                    prefix = 'LARGE ERROR: '
                    if line.startswith(prefix):
                        f.write(line[len(prefix):])
        return True
    return False

def getscores_for_dir(dirname, key=None, task=(0,None)):
    if key is None:
        if not os.path.exists(dirname+'/results'):
            if not create_results(dirname):
                return
        scores = np.array(list(get_simple_scores(dirname+'/results', task[1])))
    else:
        scores = np.array(list(get_scores_for_key(dirname+'/log0', key, task[0])))
    if not scores.size:
        return None
    locs, vals = scores.T
    df = pd.Series(index=locs, data=vals)
    return df


def getscores_for_fileset(filenames, key=None, task=(0,None)):
    all_series = []
    for dirname in filenames:
        df = getscores_for_dir(dirname, key, task)
        if df is None or len(df) < 2:
            continue
        all_series.append(df)
    all_series = [a.groupby(level=0).first() for a in all_series] # XXX legacy hack
    data = pd.DataFrame(all_series).T
    data.loc[data.first_valid_index()] = data.loc[data.first_valid_index()].fillna(1)
    data = data.interpolate(method='nearest')
    return data

def get_name(fname):
    return '/'.join(fname.split('/')[:2])

def print_results(fname, score_pairs):
    name = get_name(fname)
    if not score_pairs:
        print name, '(none)'
        return
    locs = [x for x in score_pairs if x[1] == 0]
    first_loc = locs[0][0] if locs else None
    scores = [x[1] for x in score_pairs[-5:]]
    result = np.mean(scores)
    print '%s\t%s\t%s %s\t%s' % (name, result, np.min(scores), np.max(scores), first_loc)

def plot_start(key):
    pylab.xlabel('Steps of training')
    if key:
        pylab.ylabel(key)
    else:
        pylab.ylabel('Sequence error on large input')

def plot_results(fname, frame):
    x = frame.index
    ysets = list(frame.T.values)
    if args.smoothing > 1:
        f = lambda y: scipy.signal.savgol_filter(y, args.smoothing, 1)
    else:
        f = lambda y: y
    ysets = np.array(map(f, ysets)).T
    y = np.median(ysets, axis=1) if args.median else ysets.mean(axis=1)
    v=pylab.plot(x, y,
               label=get_name(fname),
               marker='o',
    )
    for ys in list(ysets.T):
        pylab.plot(x, ys, alpha=0.2,
                   color=v[0].get_color(),
        )
    pylab.fill_between(frame.index, ysets.min(axis=1), ysets.max(axis=1),
                       alpha=0.15, color=v[0].get_color())

    #for k in frame.columns:
    #    pylab.scatter(frame.index, frame[k].values, alpha=0.15, color=v[0].get_color())

def main2(func, files):
    for fname in files:
        with open(fname) as f:
            score_pairs = getscores(f)
        func(fname, score_pairs)

def get_tasks(key):
    if 'task' not in key:
        return [None]
    else:
        locs = key.split('=')
        index = [i for i,a in enumerate(locs) if a.endswith('task')][0]+1
        tasks = locs[index].split('-')[:-1 if index < len(locs)-1 else None]
        return tasks

def get_key(fname):
    return '/'.join(('-'.join(fname.split('-')[:-2])).split('/')[-2:])

def plot_all(func, files, key=None, taskset=None):
    d = {}
    for f in files:
        d.setdefault(get_key(f), []).append(f)

    longest_cp = os.path.commonprefix(d.keys())
    print longest_cp
    if longest_cp[-1] == '=': #prettier
        longest_cp = longest_cp.rsplit('-',1)[0]+'-'

    for k in sorted(d):
        tasks = get_tasks(k)
        for i, task in enumerate(tasks):
            if task not in taskset:
                continue
            scores = getscores_for_fileset(d[k], key, (i, task))
            if not len(scores):
                continue
            if all('do_lastout=True-' in kk for kk in d):
                k = k.replace('do_lastout=True-', '')
            func(k[len(longest_cp):] + (' (%s)' % task if task and len(tasks) > 1 else ''), scores)
    return longest_cp

if __name__ == '__main__':
    args =  parser.parse_args()
    all_tasks = sorted(set(x for file in args.files for x in get_tasks(get_key(file))))
    if args.task == 'print':
        plot_all(print_results, args.files, key=args.key)
    elif args.task == 'plot':
        global pylab
        import pylab
        for i, task in enumerate(all_tasks):
            pylab.subplot(1, len(all_tasks), i+1)
            plot_start(args.key)
            subtitle = plot_all(plot_results, args.files, key=args.key,
                                taskset = [task])
            pylab.legend(loc=0)
            pylab.ylim((0, None))
            title = args.title
            if not title:
                title = os.path.split(args.files[0])[-2]
            if subtitle:
                title += '\nCommon args: %s' % subtitle
            if len(all_tasks) > 1:
                title = '%s\n%s' % (task, title)
            pylab.title(title)
        pylab.show()
