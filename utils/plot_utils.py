import os
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC


def draw_individual_plots(plot_dict, res_path):
    
    for name in ['train', 'val', 'test']: 
        x, y, p = plot_dict[name][0], plot_dict[name][1], plot_dict[name][2], 
        epoch = plot_dict['epoch']
        
        x = TSNE(n_components=2).fit_transform(x)
        df = pd.DataFrame(columns=['x0', 'x1', 'label', 'prediction'])
        df['x0'], df['x1'], df['label'], df['prediction'] = x[:, 0], x[:, 1], y, p

        sns.pairplot(x_vars=['x0'], y_vars=['x1'], data=df, hue='label', height=5)
        plt.savefig(os.path.join(res_path, 'imgs', f'{name}_{epoch}_label.png'))
        plt.close()

        sns.pairplot(x_vars=['x0'], y_vars=['x1'], data=df, hue='prediction', height=5)
        plt.savefig(os.path.join(res_path, 'imgs', f'{name}_{epoch}_prediction.png'))
        plt.close()


def draw_train_test_plot_backup(plot_dict, res_path):

    x = np.concatenate([plot_dict['train'][0], plot_dict['test'][0]])
    y = np.concatenate([plot_dict['train'][1], plot_dict['test'][1] + 2])
    p = np.concatenate([plot_dict['train'][2], plot_dict['test'][2] + 2])
    epoch = plot_dict['train'][3]

    x = TSNE(n_components=2, random_state=3333).fit_transform(x)
    df = pd.DataFrame(columns=['x0', 'x1', 'y', 'p'])
    df['x0'], df['x1'], df['y'], df['p'] = x[:, 0], x[:, 1], y, p

    colors = sns.color_palette("tab10", 4)
    palette = {'train_0': colors[0], 'train_1': colors[2], 'test_0': colors[1], 'test_1': colors[3]}
    markers = {'train_0': 'o', 'train_1': 'o', 'test_0': 'x', 'test_1': 'x'}

    def plot(col, name):    
        conditions = [df[col] == 0, df[col] == 1, df[col] == 2, df[col] == 3]
        df[name] = np.select(conditions, ['train_0', 'train_1', 'test_0', 'test_1'])
        sorted_df = df.sort_values(by='label', ascending=True)
        sns.pairplot(x_vars=['x0'], y_vars=['x1'], data=sorted_df, hue=name, palette=palette, height=5)
        plt.savefig(os.path.join(res_path, 'imgs', f'{epoch}_{name}.png'))
        plt.close()

    plot(col='y', name='label')        
    plot(col='p', name='prediction')    


def draw_train_test_plot(plot_dict, res_path):

    x = np.concatenate([plot_dict['train'][0], plot_dict['test'][0]])
    y = np.concatenate([plot_dict['train'][1], plot_dict['test'][1] + 2])
    p = np.concatenate([plot_dict['train'][2], plot_dict['test'][2] + 2])
    epoch = plot_dict['epoch']

    x = TSNE(n_components=2, random_state=3333).fit_transform(x)
    df = pd.DataFrame(columns=['x0', 'x1', 'y', 'p'])
    df['x0'], df['x1'], df['y'], df['p'] = x[:, 0], x[:, 1], y, p

    labels = ['train_0', 'train_1', 'test_0', 'test_1']
    colors = sns.color_palette("tab10", 4)
    palette = {'train_0': colors[0], 'train_1': colors[0], 'test_0': colors[1], 'test_1': colors[1]}
    markers = {'train_0': 'o', 'train_1': 's', 'test_0': 'o', 'test_1': 's'}

    def plot(col, name):
        fig = plt.figure(figsize=(15, 15))
        for i in [0, 2, 1, 3]:
            mi, ci = markers[labels[i]], palette[labels[i]]
            x0, x1 = df[df[col] == i]['x0'], df[df[col] == i]['x1']
            if mi == 'o':
                plt.scatter(x0, x1, s=100, facecolor='none', edgecolors=ci, marker=mi, label=labels[i]) #, alpha=0.9)
            else:
                plt.scatter(x0, x1, s=100, color=ci, marker=mi, label=labels[i])
                
        plt.legend(prop={'size': 25})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(os.path.join(res_path, 'imgs', f'{epoch}_{name}.png'))
        plt.close()

    plot(col='y', name='label')        
    plot(col='p', name='prediction')    
    
    
def draw_train_test_center_plot(plot_dict, res_path):
                          
    x = np.concatenate([plot_dict['train'][0], plot_dict['test'][0]])
    y = np.concatenate([plot_dict['train'][1], plot_dict['test'][1] + 2])
    p = np.concatenate([plot_dict['train'][2], plot_dict['test'][2] + 2])
    epoch = plot_dict['epoch']
    c = plot_dict['center']
    c_dim = c.shape[0]
    
    x = np.concatenate([x, c])
    x = TSNE(n_components=2, random_state=3333).fit_transform(x)
    df = pd.DataFrame(columns=['x0', 'x1', 'y', 'p'])
    df['x0'], df['x1'], df['y'], df['p'] = x[:-c_dim, 0], x[:-c_dim, 1], y, p

    labels = ['train_0', 'train_1', 'test_0', 'test_1']
    colors = sns.color_palette("tab10", 4)
    palette = {'train_0': colors[0], 'train_1': colors[0], 'test_0': colors[1], 'test_1': colors[1]}
    markers = {'train_0': 'o', 'train_1': 's', 'test_0': 'o', 'test_1': 's'}

    def plot(col, name):
        fig = plt.figure(figsize=(15, 15))
        for i in [0, 2, 1, 3]:
            mi, ci = markers[labels[i]], palette[labels[i]]
            x0, x1 = df[df[col] == i]['x0'], df[df[col] == i]['x1']
            if mi == 'o':
                plt.scatter(x0, x1, s=100, facecolor='none', edgecolors=ci, marker=mi, label=labels[i])
            else:
                plt.scatter(x0, x1, s=100, color=ci, marker=mi, label=labels[i])
        
        color = ['g', 'r', 'b', 'm', 'c', 'y']
        for i in range(c_dim, 0, -1):
            print(x[-i][0], x[-i][1])
            plt.scatter(x[-i][0], x[-i][1], s=800, color=color[i], marker='X', label=f'Center {i}')
   
        plt.legend(prop={'size': 25})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(os.path.join(res_path, 'imgs', f'{epoch}_{name}.png'))
        plt.close()

    plot(col='y', name='label')        
    plot(col='p', name='prediction')    

    
def draw_roc_curve(input_x, input_y, label=1, title=''):

    x, y = input_x[input_y < 2], input_y[input_y < 2].reshape(-1, 1)
    y = OneHotEncoder(sparse=False).fit_transform(y)
    n_classes = y.shape[1]
    random_state = np.random.RandomState(0)

    # shuffle and split training and test sets
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.4, random_state=0)
    
    # learn to predict each class against the other
    classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=random_state))
    y_score = classifier.fit(train_x, train_y).decision_function(test_x)

    # compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(5, 5))
    plt.plot(fpr[label], tpr[label], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[label])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
