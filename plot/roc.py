import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import os
import Tools.Assessment as Assessment
import mpl_toolkits.axisartist.axislines as axislines

if __name__ == "__main__":
    # The path of the original result of the roc curve
    path = 'result/total/roc.pe-6.txt'

    labels = [
        'GNS',
        'SiNW',
        'GNS+SiNW',
    ]

    # colors = ['aqua','g', 'darkorange' ,'b','r','y','darkorange','pink','g']
    # colors = ['#e41a1c','#377eb8','#ffff33','#ff7f00','#984ea3','#a65628',]
    colors = ['#FE9E37','#EF595A','#9A77B8','#EEE999','#62A3CB','#72BF5A']
    # colors = ['#FE9E37','#62A3CB','#9A77B8','#EF595A','#72BF5A',]
    # colors = ["#62A3CB","#72BF5A","#EF595A","#FE9E37"]
    # colors = ['#a65628', '#ffff33','#377eb8','#e41a1c','#984ea3' ]
    # colors = ["#62A3CB","#72BF5A","#EF595A","#FE9E37"]

    ##############################################################################
    plt.rc('font',family='Times New Roman',size='18')
    plt.rcParams['figure.figsize'] = (6.5, 6.0)
    fig, ax = plt.subplots()
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(len(labels)):
            line_true = lines[2*i].replace('\n', '')
            line_score = lines[2*i+1].replace('\n', '')
            y_true = line_true.split(' ')
            y_score = line_score.split(' ')
            print(y_score)
            Assessment.plot_roc(np.array(y_true, dtype=np.int32), 
                                np.array(y_score, dtype=np.float),
                                labels[i], colors)
    plt.legend()
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='18')
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    plt.xlabel('100 - Specificity (%)')
    plt.ylabel('Sensitivity (%)')
    plt.tight_layout()
    plt.savefig(path.replace('.txt','.jpg'), dpi=300)
    plt.show()