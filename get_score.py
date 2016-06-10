import fileinput

import sys
import numpy as np
import pandas as pd
import argparse
import glob
import scipy.signal

parser = argparse.ArgumentParser(description='Get scores')

parser.add_argument("--task", type=str, default='plot')
parser.add_argument("--title", type=str, default='')
parser.add_argument("--median", action='store_true')
parser.add_argument("--smoothing", type=int, default='1')
parser.add_argument('files', type=str, nargs='+',
                    help='Log files to examine')

def getscores(f):
    lst = []
    for line in f:
        # New style: all relevant
        if True or 'LARGE ERROR' in line:
            loc, val = line.split()[-2:]
            try:
                lst.append((int(loc), float(val)))
            except ValueError:
                break
    return lst

def getscores_for_fileset(filenames):
    all_series = []
    for fname in filenames:
        if not fname.endswith('/results'):
            fname += '/results'
        with open(fname) as f:
            scores = getscores(f)
            if not scores:
                continue
            locs, vals = np.array(scores).T
            df = pd.Series(index=locs, data=vals)
            all_series.append(df)
    data = pd.DataFrame(all_series).T
    if not len(data):
        return data
    data.values[0] = 1
    data = data.interpolate()
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

def plot_start():
    global pylab
    import pylab
    pylab.xlabel('Steps of training')
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

def main(func, files):
    def get_key(fname):
        return ('-'.join(fname.split('-')[:-2])).split('/')[-1]
    d = {}
    for f in files:
        d.setdefault(get_key(f), []).append(f)
    for k in d:
        scores = getscores_for_fileset(d[k])
        if not len(scores):
            continue
        func(str(k), scores)

if __name__ == '__main__':
    args =  parser.parse_args()
    if args.task == 'print':
        main(print_results, args.files)
    elif args.task == 'plot':
        plot_start()
        main(plot_results, args.files)
        pylab.legend(loc=0)
        pylab.ylim((0, None))
        title = args.title
        if not title:
            title = os.path.split(args.files[0])[-2]
        pylab.title(title)
        pylab.show()
