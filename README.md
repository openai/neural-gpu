**Status:** Archive (code is provided as-is, no updates expected)

Code for the Neural GPU model originally described in
[[http://arxiv.org/abs/1511.08228]].


Running experiments
===================

Running one instance
--------------------

The following would use 256 filters to train on binary multiplication,
then 4-ary, then decimal:
```
python neural_gpu_trainer.py --nmaps=256 --task=bmul,qmul,mul --progressive_curriculum=5
```

My typical invocation is something like

```
  CUDA_VISIBLE_DEVICES=0 python neural_gpu_trainer.py --random_seed=0 --max_steps=200000 --forward_max=201 --nmaps=256 --task=bmul,qmul,mul --time_till_eval=4 --progressive_curriculum=5 --train_dir=../logs/August-12-curriculum/forward_max=201-nmaps=256-task=bmul,qmul,mul-progressive_curriculum=5-random_seed=0
```

The tests on decimal carry were done using invocations like the following:
```
  CUDA_VISIBLE_DEVICES=0 neural_gpu_trainer.py --train_dir=../logs/run1 --random_seed=1 --max_steps=100000 --forward_max=201 --nmaps=128 --task=add --time_till_eval=4 --time_till_ckpt=1
```

You can find a list of options, and their default values, in `neuralgpu/trainer.py`.

Examining results
=================

Loading and examining a model
-----------------------------

`examples/examples_for_loading_model.py` gives a simple instance of loading a
model and running it on an instance.

Plotting results
----------------

Something like `python plots/get_pretty_score.py cachedlogs/*/*task=bmul,qmul,mul-*` works.  There are a lot of options to make it prettier (renaming stuff, removing some runs, changing titles, reordering, etc.).  For example, one of my plots was made with

```
python get_pretty_score.py cachedlogs/A*/*256*[=,]mul-* --titles '256 filters|' --title 'Decimal multiplication is easier with curriculum' --task mul --remove_strings='|-progressive_curriculum=5' --exclude='layer|progressive' --order '4,2,1,3' --global-legend=1
```

Requirements
============

* TensorFlow (see tensorflow.org for how to install)
* Matplotlib for Python (sudo apt-get install python-matplotlib)
* joblib

Credits
=======

Original code by Lukasz Kaiser (lukaszkaiser).  Modified by Eric Price
(ecprice)
