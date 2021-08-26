import sys
sys.path.append("../")
import Tools.Assessment as Assessment
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

def colormap():
    cdict = ['#FFFFFF','#FDBE85', '#FDA560','#FD8D3C']
    return colors.LinearSegmentedColormap.from_list('my',cdict)


if __name__ == "__main__":
    # The path of the original result of the confusion matrix (saved in '.txt' format)
    fnames = ['plot/data/batch/cm-single-0.5-S2-new', 'plot/data/batch/cm-single-0.8-S2-new']

    tags = ['HC','HCC','NSCLC','PAAD','CRC','GC','PTC']
    nums = [0, 0, 0, 0, 0, 0, 0]
    for index, fname in enumerate(fnames):
        overall_accu = 0
        cm = np.loadtxt(fname+'.txt')

        tagLen = len(tags)
        sum_cm = np.zeros((tagLen,tagLen))
        for i in range(1):
            sum_cm += cm[i*tagLen:(i+1)*tagLen,:]
        # sum_cm /= 10
        
        # Calu overall accu
        sum_cm = sum_cm / np.sum(sum_cm, axis=1)
        sum_cm[np.isnan(sum_cm)] = 0
        print(sum_cm)
        for i in range(len(tags)):
            overall_accu += nums[i] * sum_cm[i, i]
        
        overall_accu = overall_accu / sum(nums) * 100
        Assessment.plot_Matrix(cm, tags , True, None, fname, colormap(), overall_accu)
        # Assessment.plot_Matrix(cm, tags , True, None, fname,  plt.cm.Blues)
        # Assessment.plot_Matrix(sum_cm, tags , True, None, fname, plt.cm.Blues)