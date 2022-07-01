"""
Plot results
"""




#==============#
# Imports
#==============#
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import random
import itertools
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(1, './')

from module import utils, models

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
tf.get_logger().setLevel('ERROR')

from sklearn import metrics
from sklearn.model_selection import KFold





# Accuracy, MCC bar plot
#================================#
path = './experiment/data/training/performance'
accs_train = pd.read_csv(f'{path}/accuracy_validation.csv', index_col=0).values.reshape(-1)
mccs_train = pd.read_csv(f'{path}/mcc_validation.csv', index_col=0).values.reshape(-1)
rhos_train = pd.read_csv(f'{path}/rho_testing.csv', index_col=0).iloc[:,0].values.reshape(-1)

path = './experiment/data/finetuning/performance_1e-6'
accs_fine = pd.read_csv(f'{path}/accuracy_validation.csv', index_col=0).values.reshape(-1)
mccs_fine = pd.read_csv(f'{path}/mcc_validation.csv', index_col=0).values.reshape(-1)
rhos_fine = pd.read_csv(f'{path}/rho_testing.csv', index_col=0).iloc[:,0].values.reshape(-1)

accs = [np.mean(accs_train), np.mean(accs_fine)]
mccs = [np.mean(mccs_train), np.mean(mccs_fine)]
rhos = [np.mean(rhos_train), np.mean(rhos_fine)]

accs_std = np.array([np.std(accs_train), np.std(accs_fine)]) / np.sqrt(3)
mccs_std = np.array([np.std(mccs_train), np.std(mccs_fine)]) / np.sqrt(3)
rhos_std = np.array([np.std(rhos_train), np.std(rhos_fine)]) / np.sqrt(3)


fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'18'}
legend_font = {'family':fnt, 'size':'13'}
label_font = {'family':fnt, 'size':'20'}
plt.rcParams['figure.figsize'] = [6,5]

gap = 5
plt.bar([0, gap+0], accs, yerr=accs_std, color='#06592A', edgecolor='black', 
         width=0.95, capsize=3, label='Accuracy')
plt.bar([1, gap+1], mccs, yerr=mccs_std, color='#226E9C', edgecolor='black', 
         width=0.95, capsize=3, label='MCC')
plt.bar([2, gap+2], rhos, yerr=rhos_std, color='#8F003B', edgecolor='black', 
        width=0.95, capsize=3, label='Spearman R')
plt.xticks([1, 6], ['Training', 'Fine-tuning'], **ticks_font)
plt.yticks(**ticks_font)
plt.ylim((0,1.19))
plt.legend(prop=legend_font, loc='upper right')
plt.savefig('performance.pdf')





# HMM correlation with each dataset
from deepPETase import DeepPETase, utils
dp = DeepPETase()
label = '/Users/jgado/Dropbox/research/projects/deepPETase_project/deepPETase_design/experiment/data/preprocessing/label_sequences.fasta'
dp.alignWithHMM(label, threshold=0, outdir='temp/hmmdir')
df = utils.parse_hmm_tabout('temp/hmmdir/tab_output.txt')

dflabel = pd.read_excel('experiment/data/labels/datasets.xlsx', index_col=0)
labels = [item.replace(',','').replace(' ', '_') for item in dflabel.index]
hmm_rho = {}

for dataset in labels:
    dfeach = pd.read_csv(f'experiment/data/labels/activity_data/{dataset}.csv', 
                         index_col=0)
    dfeach = dfeach.dropna()
    locs = [list(df.index).index(item) for item in dfeach.index]
    dfsel = df.iloc[locs,:]
    rho = spearmanr(dfsel['Score'].values, dfeach['Activity'].values)[0]
    hmm_rho[dataset] = rho
hmm_rho = pd.Series(hmm_rho)
print(hmm_rho.mean())    
print(hmm_rho.median())




# Rho for each dataset
#========================#

fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'14'}
legend_font = {'family':fnt, 'size':'12'}
label_font = {'family':fnt, 'size':'14'}
plt.rcParams['figure.figsize'] = [8,8]



path = './experiment/data/training/performance'
rhos_train = pd.read_csv(f'{path}/rho_heldout.csv', index_col=0)
path = './experiment/data/finetuning/performance/1e-07'
rhos_fine = pd.read_csv(f'{path}/rho_heldout.csv', index_col=0)

rhos_fine = rhos_fine.sort_values('0', ascending=False)
rhos_train = rhos_train.reindex(rhos_fine.index)
hmm_rho = hmm_rho.reindex(rhos_fine.index)

# Plot means
y = hmm_rho.mean()
plt.errorbar(y, -2, color='lightpink', xerr=None, fmt='o', capsize=0, markersize=10, 
             label='Hidden Markov model')
y, yerr = rhos_train.mean()
plt.errorbar(y, -2, color='grey', xerr=yerr, fmt='o', capsize=0, markersize=10, 
             label='Separate top model')
y, yerr = rhos_fine.mean()
plt.errorbar(y, -2, color='darkblue', xerr=yerr, fmt='o', capsize=0, markersize=10, 
             label='End-to-end fine-tuning')
plt.hlines(y=-2, xmin=-1, xmax=y, color='grey', linestyle='--', alpha=0.5, linewidth=1)


for i,name in enumerate(rhos_fine.index):
    y = hmm_rho.loc[name]
    plt.errorbar(y, i-0, color='pink', xerr=None, fmt='o', capsize=0, markersize=10, 
                 zorder=1, alpha=0.5)
    y1, yerr = rhos_fine.loc[name,:]
    plt.errorbar(y1, i-0, color='darkblue', xerr=yerr, fmt='o', capsize=0, markersize=10, 
                 zorder=2)
    y2, yerr2 = rhos_train.loc[name,:]
    plt.errorbar(y2, i+0, color='grey', xerr=yerr2, fmt='o', capsize=0, markersize=10,
                 zorder=1)
    plt.hlines(y=i, xmin=-1, xmax=(max(y, y1, y2)), color='grey', linestyle='--', alpha=0.5, linewidth=1)




yticks = [item.split('_') for item in rhos_fine.index]
yticks = [' '.join(item[:-1]) + ', ' + item[-1] for item in yticks]
yticks = ['Average'] + yticks
plt.xticks(**ticks_font)
plt.yticks([-2] + list(np.arange(len(rhos_train),step=1)), yticks, **ticks_font)
plt.xlim((-1,1.2))
plt.ylim((-3,26))
plt.xlabel('Spearman R', **label_font)
plt.legend(loc='best', prop=legend_font)
plt.tight_layout()
plt.savefig('temp/plots/final_spearmar_all.pdf')
    









# Box plots


# Font
fnt = 'Arial'
ticks_font = {'fontname':fnt, 'size':'12'}
legend_font = {'family':fnt, 'size':'12'}
label_font = {'family':fnt, 'size':'14'}
plt.rcParams["figure.figsize"] = [4,3]
#plt.rcParams['grid.alpha'] = 0.5





# Boxplot specifications
#positions = np.arange(9) * (len(mlkeys) + 3) + i
for i,key in enumerate(['HMM', 'Separate top model', 'End-to-end fine-tuning']):
    positions = [i]
    color = 'lightpink' if i==0 else 'lightgrey' if i==1 else 'lightblue'
    

    meanprops = {'marker':'o',
                'markerfacecolor':color,
                'markeredgecolor':'black',
                'markersize':5.0,
                'linewidth':1.0}
    medianprops = {'linestyle':'-',
                   'linewidth':1.0,
                   'color':'black'}
    boxprops = {'facecolor':color,
                'color':'black',
                'linewidth':1.0}
    flierprops = {'marker':'o',
                  'markerfacecolor':'black',
                  'markersize':1,
                  'markeredgecolor':'black'}
    whiskerprops = {'linewidth':1.0}
    capprops = {'linewidth':1.0}
    
    # Plot the boxplot
    data = hmm_rho if i==0 else rhos_train.iloc[:,0] if i == 1 else rhos_fine.iloc[:,0]
    _ = plt.boxplot(data, 
                    positions=positions, 
                    widths=0.5,#(1, 1, 1),
                    #widths=None,
                    whis=(0,100),               # Percentiles for whiskers
                    #whis=1.5,
                    showmeans=False,             # Show means in addition to median
                    patch_artist=True,          # Fill with color
                    meanprops=meanprops,        # Customize mean points
                    medianprops=medianprops,    # Customize median points
                    boxprops=boxprops,
                    showfliers=False,            # Show/hide points beyond whiskers            
                    flierprops=flierprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    zorder=1)

    # Scatter plot
    xs = np.random.normal(i, 0.02, size=len(data))
    plt.scatter(xs, data, color='grey', edgecolor='black', zorder=2, s=20, alpha=1.0)

#plt.xlim((-1,1))
plt.yticks(**ticks_font)
plt.xticks([0,1,2], ['HMM', 'Separate\ntop model', 'End-to-end\nfine-tuning'], **ticks_font)
plt.ylabel('Spearman R', **label_font)
plt.tight_layout()
plt.savefig('temp/plots/boxwhisker_spearmanr.pdf')




# Erickson scatter
from temp import adhoc

path = 'experiment/data/finetuning/performance/1e-07/predictions'
df = pd.read_csv(f'{path}/Erickson_et_al_2022.csv', index_col=0)
adhoc.plot_scatter(df['ypred'], df['ylog'], scatter_size=50, savepath='temp/plots/erickson.pdf',
                   ylabel='log(1+Activity)')

path = 'experiment/data/finetuning/performance/1e-07/predictions'
df = pd.read_csv(f'{path}/Chen_et_al_2021.csv', index_col=0)
adhoc.plot_scatter(df['ypred'], df['ylog'], scatter_size=50, savepath='temp/plots/Chen.pdf',
                   ylabel='log(1+Activity)')


path = 'experiment/data/finetuning/performance/1e-07/predictions'
df = pd.read_csv(f'{path}/Sonnendecker_et_al_2021.csv', index_col=0)
adhoc.plot_scatter(df['ypred'], df['ylog'], scatter_size=50, savepath='temp/plots/Sonnendecker.pdf',
                   ylabel=None)




df = pd.read_csv(f'{path}/Zeng_et_al_2022.csv', index_col=0)
adhoc.plot_scatter(df['ypred'], df['ytrue'], scatter_size=50, savepath='temp/plots/Zeng.pdf', 
                   ylabel=None)


df = pd.read_csv(f'{path}/Cui_et_al_2021.csv', index_col=0)
adhoc.plot_scatter(df['ypred'], df['ytrue'], scatter_size=50, savepath='temp/plots/Cui.pdf',
                   ylabel=None)



df = pd.read_csv(f'{path}/Bell_et_al_2022.csv', index_col=0)
adhoc.plot_scatter(df['ypred'], df['ytrue'], scatter_size=100, savepath='temp/plots/Bell.pdf',
                   ylabel=None)



df = pd.read_csv(f'{path}/Nakamura_et_al_2021.csv', index_col=0)
adhoc.plot_scatter(df['ypred'], df['ytrue'], scatter_size=50, savepath='temp/plots/Nakamura.pdf')


df = pd.read_csv(f'{path}/Sonnendecker_et_al_2021.csv', index_col=0)
adhoc.plot_scatter(df['ypred'], df['ytrue'], scatter_size=50, savepath='temp/plots/Sonnendecker.pdf',
                   ylabel=None)


df = pd.read_csv(f'{path}/Then_et_al_2016.csv', index_col=0)
adhoc.plot_scatter(df['ypred'], df['ytrue'], scatter_size=50, savepath='temp/plots/Then.pdf',
                   ylabel=None)


                   
path = 'experiment/data/finetuning/performance/predictions/Chen_et_al_2021.csv'
df = pd.read_csv(path, index_col=0)
adhoc.plot_scatter(df['ypred'], df['ytrue'], scatter_size=50, savepath='Chen.pdf')

path = 'experiment/data/finetuning/performance/predictions/Sonnendecker_et_al_2021.csv'
df = pd.read_csv(path, index_col=0)
adhoc.plot_scatter(df['ypred'], df['ytrue'], scatter_size=50, savepath='Sonnendecker.pdf')



                      
                   
                   
                   
                   
                   