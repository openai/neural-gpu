import pandas
import pylab
from matplotlib import rc
import matplotlib.ticker as mtick

rc('font',  size='9')
rc('axes', labelsize='large')
rc('lines', linewidth=3)

supsize=12
titlesize=9
legsize=8

if __name__ != '__main__':
    pylab.ion()

vertical = True
#vertical = False
#scp 6.cirrascale.sci.openai.org:models/neural_gpu/*.csv .


def make_plot(csv_name, kws1={}):
    data = pandas.read_csv(csv_name)
    data = data.set_index(data.columns[0])
    data.index.name = 'Carries'
    data = data * 1. / data.values.max()
    data = data[:60]

    my_kws = dict(marker='o', ms=5)
    k1 = my_kws.copy()
    k1.update(kws1)
    #k2 = my_kws.copy()
    #k2.update(label="Result barely carries")
    #k2.update(kws2)
    #pylab.plot(data['False'], **k1)
    #pylab.plot(data['True'], **k2)
    pylab.plot((data['True'] + data['False'])/2, **k1)


if vertical:
    pylab.figure(figsize=(4,4))
    orientation = (2,1)
else:
    pylab.figure(figsize=(6,3))
    orientation = (1,2)

pylab.subplot(*(orientation+(1,)))
make_plot('csv/carry_errors_big.csv',
          dict(label='Train on random examples'),
         )
make_plot('csv/carry_errors_big.baddt.csv',
          dict(label='Train with some hard examples'),
         )
#pylab.xlabel("Number of carries $k$")
#pylab.legend(loc=0, prop={'size': legsize})
pylab.gca().yaxis.set_major_formatter(mtick.FuncFormatter(
    lambda x, pos: '% 2d%%' % (x*100)))
if not vertical:
    pylab.xlabel("Number of carries $k$")
pylab.ylabel("Error rate")
pylab.title("Binary", size=titlesize)
pylab.locator_params(axis='y',nbins=4)#

pylab.subplot(*(orientation+(2,)))
make_plot('csv/carry_errors_add_large.csv',
          dict(label='Train on random examples'),
         )
make_plot('csv/carry_errors_addt_large.csv',
          dict(label='Train with some hard examples'),
         )
pylab.xlabel("Number of carries $k$")
#pylab.legend(loc=0, prop={'size': legsize})
if vertical:
    pylab.ylabel("Error rate")
pylab.gca().yaxis.set_major_formatter(mtick.FuncFormatter(
    lambda x, pos: '% 2d%%' % (x*100)))
pylab.title("Decimal", size=titlesize)
pylab.suptitle("Additions with long carries", size=supsize)
pylab.locator_params(axis='y',nbins=4)#
pylab.gcf().legend(*pylab.gca().get_legend_handles_labels(), loc='lower center', ncol=1, labelspacing=0)
if vertical:
    pylab.tight_layout(rect=[0, 0.14, 1, 0.95])
else:
    pylab.tight_layout(rect=[0, 0.18, 1, 0.93])
if vertical:
    pylab.savefig('../neuralgpu_paper/fig_carries_all_vertical.pdf')
else:
    pylab.savefig('../neuralgpu_paper/fig_carries_all_horizontal.pdf')
