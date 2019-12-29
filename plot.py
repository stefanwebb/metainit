from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path, pickle, math
import numpy as np
import matplotlib
#matplotlib.use('pdf')
matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

plt.style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
#matplotlib.rc('font', family='Latin Modern Roman')
#mathtext.fontset
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

viridis = plt.get_cmap('viridis').colors
viridis = [viridis[i] for i in [100, 240, 150, 0]]
tab20b = plt.get_cmap('tab20b').colors
colors = [None for _ in range(4)]
colors[0] = viridis[0]
colors[1] = plt.get_cmap('Paired').colors[3]
colors[2] = tab20b[13]
colors[3] = (0.3, 0.3, 0.3)

def cm2inch(value):
  return value/2.54

# Open the summary because we want to pick out certain properties!
#summary = pickle.load(open(f'../data/collisionDetection/summary.pickle', "rb" ))
#with open(f'../results/mnist/extracted_exp_results.pickle', 'wb') as handle:
    #pickle.dump({'lg_ps':lg_ps, 'naive_lg_ps':naive_lg_ps, 'naive_counts':naive_counts, 'naive_totals':naive_totals}, handle, protocol=pickle.HIGHEST_PROTOCOL)

def plot_weights(weights):
  """
  Plot layer weights
  """
  #results = pickle.load(open('../results/mnist/extracted_exp_results.pickle', 'rb'))

  
  fig = plt.figure(figsize=(cm2inch(8.0), cm2inch(6.0)))
  #fig.suptitle(f'CollisionDetection, Property {property_id}, lg(p)={brute_val}')
  ax = fig.add_subplot(1, 1, 1, xlabel=r'layer', ylabel=r'norm')  #, ylim=(0,15))
  #ax.hlines(y=0, linestyle='--', xmin=xs[0], xmax=xs[-1], linewidth=0.75, color='black') #color='red',
  #ax.hlines(y=-100, color='black', xmin=xs[0], xmax=xs[-1], linestyle='--', linewidth=0.75)

  for idx, (label, y) in enumerate(weights):
    x = list(range(len(y)))
    ax.plot(x, y, color=colors[idx], linestyle='-', marker='.', label=label)

  ax.legend()
  ax.xaxis.set_tick_params(width=0.5)
  ax.yaxis.set_tick_params(width=0.5)
  ax.spines['left'].set_linewidth(0.5)
  ax.spines['bottom'].set_linewidth(0.5)
  sns.despine()

  fig.savefig(f'layer_weights.svg', bbox_inches='tight')
  plt.close(fig)
