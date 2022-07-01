"""
Adhoc functions for training semi-supervised models
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, './')
from module import utils

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder







NORMALIZED_CSV = './experiment/preprocessing/data/labeled_data/assay/'\
                 'activity_normalized.csv'
SAMPLE_WEIGHT_CSV =  './experiment/preprocessing/data/labeled_data/assay/'\
                      'sample_weights.csv'
CLUSTER_FILE =     './experiment/preprocessing/data/labeled_data/sequence/cluster.txt' 
SEQUENCE_FILE = './experiment/preprocessing/data/labeled_data/sequence/'\
                'hmm_sequences.fasta' 
                



                      
def get_lcocv_ids():
    '''Return ids for splitting activity data into training and testing folds with
    leave-cluster-out cross validation'''
    
    cl_info, cl_data = utils.read_fasta(CLUSTER_FILE)
    cl_data = [eachdata.split(', ') for eachdata in cl_data]
    heads_all, _ = utils.read_fasta(SEQUENCE_FILE)
    heads_all = np.array(heads_all)
    df_activity = pd.read_csv(NORMALIZED_CSV, index_col=0)
    petcans = df_activity['petcan']
    trainids, testids, testdata = [], [], []
    
    for i in range(len(cl_data)):
        testlocs = np.argwhere(petcans.isin(cl_data[i]).values).reshape(-1)
        testids.append(testlocs)
        testdata.extend(testlocs)
        trainlocs = np.argwhere(~petcans.isin(cl_data[i]).values).reshape(-1)
        trainids.append(trainlocs)
        
    testdata = df_activity.iloc[testdata,:]
    testdata.index = range(len(testdata))
    
    return trainids, testids, testdata
    

    



def get_loocv_ids():
    '''Return ids for splitting activity data into training and testing folds with
    leave-one-sequence-out cross validation'''

    heads_all, _ = utils.read_fasta(SEQUENCE_FILE)
    heads_all = np.array(heads_all)
    df_activity = pd.read_csv(NORMALIZED_CSV, index_col=0)
    petcans = df_activity['petcan']
    trainids, testids, testdata = [], [], []
    
    for petcan in petcans.unique():
        testlocs = np.argwhere((petcans == petcan).values).reshape(-1)
        testids.append(testlocs)
        testdata.extend(testlocs)
        trainlocs = set(range(len(df_activity))) - set(testlocs)
        trainlocs = np.array(list(trainlocs))
        trainids.append(trainlocs)
        
    testdata = df_activity.iloc[testdata,:]
    testdata.index = range(len(testdata))
    
    return trainids, testids, testdata






def get_kfold_ids():
    
    heads_all, _ = utils.read_fasta(SEQUENCE_FILE)
    heads_all = np.array(heads_all)
    df_activity = pd.read_csv(NORMALIZED_CSV, index_col=0)
    petcans = df_activity['petcan']
    trainids, testids, testdata = [], [], []
    
    foldnames = {0: ['TfCut'] + \
                 [f'PETcan{x}' for x in \
                  [701, 702, 703, 704, 705, 706, 707, 708, 709, 711, 714, 715, 716, 717]],
            1: [f'PETcan{x}' for x in \
                [206, 204, 406, 503, 208, 401, 215, 306, 307, 102, 606]],
            2: [f'PETcan{x}' for x in \
                [608, 202, 209, 214, 305, 604, 301, 101, 611, 610, 211]],
            3: ['LCC_WT', 'LCC_ICCG', 'LCC_WCCG'] + [f'PETcan{x}' for x in \
                [403, 402, 504, 601, 602, 501, 407]],
            4: ['IsPETase_DM', 'IsPETase_WT'] + [f'PETcan{x}' for x in \
                [605, 103, 409, 410, 412, 713, 710, 712, 405]]}

    
    for i in foldnames.keys():
        testlocs = np.argwhere(petcans.isin(foldnames[i]).values).reshape(-1)
        testids.append(testlocs)
        testdata.extend(testlocs)
        trainlocs = np.argwhere(~petcans.isin(foldnames[i]).values).reshape(-1)
        trainids.append(trainlocs)
        
    testdata = df_activity.iloc[testdata,:]
    testdata.index = range(len(testdata))
    
    return trainids, testids, testdata






def get_activity_data(return_names=False):
    '''Return sequence data for downstream supervised learning in the same order as 
    activity data'''
    
    seqdict = utils.read_fasta(SEQUENCE_FILE, return_as_dict=True)
    df_activity = pd.read_csv(NORMALIZED_CSV, index_col=0)
    petcans = df_activity['petcan'].values
    ph_data = df_activity.ph.values.reshape(-1,1)
    hot = OneHotEncoder()
    hot.fit(ph_data)
    ph_data = hot.transform(ph_data).toarray()
    temp_data = df_activity.temp.values.reshape(-1,1)
    cond_data = np.append(ph_data, temp_data, axis=1)
    activity = df_activity.iloc[:,-1].values
    sample_weights = pd.read_csv(SAMPLE_WEIGHT_CSV).iloc[:,-1].values
    X_seq = [seqdict[seq] for seq in petcans]
    X_seq = [utils.pad_sequence(seq, maxlen=400, padtype='post') for seq in X_seq]
    X_seq = np.array([utils.one_hot_encode_sequence(seq) for seq in X_seq]) 
    if return_names: 
        return (X_seq, cond_data, activity, sample_weights, petcans)      
    else:
        return (X_seq, cond_data, activity, sample_weights)      





def get_max70_data(maxtemp=70):
    
    maxtemp = (maxtemp - 30) / (70 - 30)  # Normalize
                   
    # Activity data
    df_activity = pd.read_csv(NORMALIZED_CSV, index_col=0)
    df = pd.DataFrame(index=df_activity.columns)
    for i,name in enumerate(df_activity['petcan'].unique()):
        locs = np.logical_and((df_activity['petcan']==name).values, 
                              (df_activity['temp']>=maxtemp).values)
        sel = df_activity.iloc[locs,:].sort_values('activity', ascending=False)
        df[i] = sel.iloc[0,:]
    df = df.transpose()
    names = df['petcan'].values
    activity = df['activity'].values
    
    # Sequence data
    seqdict = utils.read_fasta(SEQUENCE_FILE, return_as_dict=True)
    X_seq = [seqdict[seq] for seq in names]
    X_seq = [utils.pad_sequence(seq, maxlen=400, padtype='post') for seq in X_seq]
    X_seq = np.array([utils.one_hot_encode_sequence(seq) for seq in X_seq]) 
    
    return X_seq, activity, names
    
    





def getSampleWeights(array, bins=10):
    freqs, borders, _ = plt.hist(array, bins=bins)
    plt.close()
    borders[-1] = borders[-1] + max(borders) # Extend right boundary to include max value
    weights = (freqs.min() / freqs) # Normalize frequency
    sample_weights = np.zeros(len(array))
    for i in range(len(borders)-1):
        low, high = borders[i], borders[i+1]
        weight = weights[i]
        locs = (array >= low).astype(int) * (array < high).astype(int)
        locs = np.argwhere(locs).reshape(-1)
        sample_weights[locs] = weight
    return sample_weights
    



def reg_metrics(ypred, ytest, weight=None):
    '''Return the spearman correlation, pearson correlation, coefficient of determination,
    and mean absolute error'''
    
    rho = spearmanr(ytest, ypred)[0]
    r = pearsonr(ytest, ypred)[0]
    r2 = r2_score(ytest, ypred, sample_weight=weight)
    mae = mean_absolute_error(ytest, ypred, sample_weight=weight)
    perf = {'rho':rho, 'r':r, 'r2':r2, 'mae':mae}
    
    return perf





PROPERTIES = {'ticks_font': {'fontname':'Arial', 'size': 12},
              'label_font': {'fontname':'Arial', 'size': 14},
              'title_font': {'fontname': 'Arial', 'size': 16}}

def plot_scatter(x, y, 
                 properties=None,
                 color='lightblue',
                 xlabel='Predicted activity', 
                 ylabel='Experimental activity',
                 title=None,
                 figsize=(5,5),
                 scatter_size=40, 
                 scatter_figwidth=0.75,
                 scatter_left=0.133,
                 scatter_linewidth=1.33,
                 scatter_alpha=0.8,
                 edgecolor='black',
                 textcoors=(0.1,0.9), 
                 bins=10,
                 savepath='./scatter.pdf'):
    '''Plot/save a scatter plot of `x` and `y` values.'''

    # Peformance
    perf = reg_metrics(x, y)
    rho, r = perf['rho'], perf['r']
    
    # Figure dimensions
    left = bottom = scatter_left
    width = height = scatter_figwidth
    histsize = 1 - left - width - 0.05
    spacing = 0.01
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing , width, histsize]
    rect_histy = [left + width + spacing, bottom, histsize, height]
    
    # Figure object
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)    
    
    # Scatter plot 
    ax.scatter(x, y, s=scatter_size, color=color, marker='o', edgecolor=edgecolor, 
                linewidth=scatter_linewidth, alpha=scatter_alpha, zorder=1)
    
    # Linear regression line
    m, b = np.polyfit(x, y, 1)
    x_ = np.array([np.min(x) - np.std(x)] + [np.max(x) + np.std(x)])
    y_ = m * x_ + b
    ax.plot(x_, y_, linestyle='--', color= 'black', alpha=0.66, zorder=2)
    
    # Label axes
    if properties is None:
        properties = PROPERTIES
        
    '''
    ax.set_xticklabels(['{:.1f}'.format(each) for each in ax.get_xticks()], 
                       **properties['ticks_font'])
    ax.set_yticklabels(['{:.1f}'.format(each) for each in ax.get_yticks()], 
                       **properties['ticks_font'])
    '''
    #ax.set_xticklabels(ax.get_xticks(), **properties['ticks_font'])
    #ax.set_yticklabels(ax.get_yticks(), **properties['ticks_font'])
    ax.set_xlabel(xlabel, **properties['label_font'])
    ax.set_ylabel(ylabel, **properties['label_font'])    
    
    # Histogram plot
    ax_histx.hist(x, color=color, bins=bins)
    ax_histy.hist(y, orientation='horizontal', color=color, bins=bins)
    ax_histx.axis(False)                
    ax_histy.axis(False)                
    
    # Annotate with text
    text = r'$\rho$={:.3f}'.format(rho) + '\nr={:.3f}'.format(r)
    textprops = dict(facecolor='None', edgecolor='grey', alpha=0.66, boxstyle='round')
    plt.text(x=textcoors[0], y=textcoors[1], s=text, transform=ax.transAxes, 
             fontsize=int(properties['ticks_font']['size']), 
             fontname=properties['ticks_font']['fontname'], verticalalignment='center',
             horizontalalignment='left', bbox=textprops)
    
    # Add title
    if title is not None:
        plt.title(title, **properties['title_font'])
             
    # Save/view
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show(); plt.close()
    





def plot_history(history, losstype='loss', dataframe=False, savepath=None):
    '''Plot training and validation history from Keras history object'''
    if dataframe:
        data = history 
    else:
        data = history.history
    fig, ax1 = plt.subplots()
    ax1.plot(data[losstype], label='Training', color='dodgerblue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(f'Training {losstype}')
    ax2 = ax1.twinx()
    ax2.plot(data[f'val_{losstype}'], label='Validation', 
             color='indianred')
    ax2.set_ylabel(f'Validation {losstype}')
    fig.legend(bbox_to_anchor=(0.86, 0.46), loc='right', ncol=1)
    plt.title(f'{losstype} in model history')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show(); plt.close()    