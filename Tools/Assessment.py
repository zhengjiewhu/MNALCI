from sklearn import metrics
import numpy as np
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.colors as colors
import math
import copy

def getMerit(y_test, y_predict):
    confusion = metrics.confusion_matrix(y_test, y_predict)
    # print(confusion)
    # print(np.sum(y_test))
    # print(y_pred_class)
    if len(confusion) == 1:
        if y_test[0] == 0:
            TP = 0
            FN = 0
            FP = 0
            TN = len(y_test)
            return 1, -1, 1
        elif y_test[0] == 1:
            TP = len(y_test)
            FN = 0
            FP = 0
            TN = 0
            return 1, 1, -1
    else:
        TP = confusion[1][1]
        FN = confusion[1][0]
        FP = confusion[0][1]
        TN = confusion[0][0]
    
    return ((TP + TN) / (TP+FP+TN+FN) , TP / (TP + FN), TN / (TN + FP))

def getAccu_proba(estimator, X, y):
    prob = estimator.predict_proba(X)
    prob_rate = prob[:,1] / prob[:,0]
    y_pred_class = np.int32(prob_rate >= 1)
    accu, _, _ = getMerit(y, y_pred_class)
    return accu

def getSens_proba(estimator, X, y):
    prob = estimator.predict_proba(X)
    prob_rate = prob[:,1] / prob[:,0]
    y_pred_class = np.int32(prob_rate >= 1)
    _, sens, _ = getMerit(y, y_pred_class)
    return sens

def getSpec_proba(estimator, X, y):
    prob = estimator.predict_proba(X)
    prob_rate = prob[:,1] / prob[:,0]
    y_pred_class = np.int32(prob_rate >= 1)
    _, _, spec = getMerit(y, y_pred_class)
    return spec

def getAccu(estimator, X, y):
    y_pred_class = estimator.predict(X)
    accu, _, _ = getMerit(y, y_pred_class)
    return accu

def getSens(estimator, X, y):
    y_pred_class = estimator.predict(X)
    _, sens, _ = getMerit(y, y_pred_class)
    return sens

def getSpec(estimator, X, y):
    y_pred_class = estimator.predict(X)
    _, _, spec = getMerit(y, y_pred_class)
    return spec

def getMatrix(y_true, y_pred, labels):
    cm = metrics.confusion_matrix(y_true, y_pred, labels)
    return cm


def plot_Matrix(cm, classes, normalize, title=None, path='cm', cmap=plt.cm.Blues, overall_accu=0):
    # plt.rcParams['figure.figsize'] = (6.0, 6.0)
    plt.rc('font',family='Times New Roman',size=13)  
    # plt.rc('xtick', labelsize=13)
    # plt.rc('ytick', labelsize=13)
    # plt.rcParams['axes.titlesize'] = 13
    # plt.rcParams['axes.titleweight'] = 13
    SMALL_SIZE = 13
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    cm_copy = copy.deepcopy(cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        str_cm = cm.astype(np.str).tolist()
        for row in str_cm:
            print('\t'.join(row))
    else:
        print('Confusion matrix, without normalization')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if math.isnan(cm[i, j]):
                cm[i, j]=0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) == 0:
                cm[i, j]=0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel="Predicted")
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)    
    if overall_accu > 0:
        ax.set_title("Overall accuracy: %d %%" % overall_accu)    
  
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)

    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), 
            rotation=45,  ha="right", va="center",
             rotation_mode="anchor",
             )

    # Loop over data dimensions and create text annotations.
    
    for i in range(cm.shape[0]):
        row_sum = 0
        for j in range(cm.shape[1]):
            cm[i, j] = int(cm[i, j]*100 + 0.5)
            row_sum += cm[i, j]
        cm[i ,i] -= row_sum - 100

    fmt = 'd' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0 and not math.isnan(cm_copy[i, j]):
                ax.text(j, i, format(int(cm[i, j]) , fmt) + '%',
                        ha="center", va="center",
                        color="black" if cm[i, j] > thresh else "black")
            elif math.isnan(cm_copy[i, j]):
                ax.text(j, i, 'N/A',
                        ha="center", va="center",
                        color="black" if cm[i, j] > thresh else "black")
            # if int(cm[i, j]*100 + 0.5) > 0:
            #     ax.text(j, i, format(int(cm[i, j]*100 + 0.5) , fmt) + '%',
            #             ha="center", va="center",
            #             color="black" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # np.savetxt(path+'.csv', cm)
    plt.savefig(path+'.jpg', dpi=300)
    plt.show()

cnt = 0
def plot_roc(y_true, y_score, label='', colors=None):
    # plt.rc('font',family='Times New Roman',size='12') 
    global cnt
    if colors == None:
        colors = ['#FE9E37','#EF595A','#9A77B8','#62A3CB','#EEE999','#72BF5A']
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    # print(thresholds)
    # roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr*100, tpr*100, color=colors[cnt],
         lw=2, label='{0} {1:0.4f} '.format(label, auc(fpr, tpr)))
    cnt += 1

def plot_multi_roc(y_true, y_score):
    n_classes = len(set(y_true))
    assert n_classes > 2
    print(list(set(y_true)))
    y_true_bin = label_binarize(y_true, classes=sorted(list(set(y_true))))
    # y_true_bin = y_true_bin.flatten()
    # y_true_bin = label_binarize(y_true_bin, classes=[i for i in range(n_classes)])
    print(y_true_bin)
    print(y_true_bin.shape)
    print(y_score.shape)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    colors = cycle(['aqua', 'pink' ,'b','r','y','darkorange','g'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(sorted(list(set(y_true)))[i], roc_auc[i]))
    plt.legend()
    plt.save()
    plt.show()

def colormap():
    cdict = ['#FFFFFF','#FDBE85', '#FDA560','#FD8D3C']
    return colors.LinearSegmentedColormap.from_list('my',cdict)