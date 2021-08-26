import sys
sys.path.append("./")
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
import os
import config
import matplotlib.pyplot  as plt
import pandas
import progressbar


def baseline_als(y, lam, p, niter=10):
    '''
    find baseline according to y
    '''
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def savitzkyGolay(x, window_length=41, polyorder=2):
    '''
    savitzkyGolay smoothing
    '''
    return savgol_filter(x, window_length, polyorder)

def align(a_l, a_r, b_l, b_r, z):
    '''
    align by 2 points
    '''
    return (a_l - a_r) * (z - b_l) / (b_l - b_r) + a_l

def main(subPosition=2, datapath=config.DATA_PATH, afterpath=config.DATA_PREPROCESS_PATH):
    totalNum = len(os.listdir(datapath))
    bar = progressbar.ProgressBar(max_value=totalNum)
    cnt = 0
    for path in os.listdir(datapath):
        cnt += 1
        bar.update(cnt)

        sType = path.split('_')[subPosition]
        # if sType not in ['A6','A8','S2','C2']:
        #     continue

        testFilePath = os.path.join(datapath, path)
        df = pandas.read_csv(testFilePath)
        matrix = df.values
        x = matrix[:,0]
        y = matrix[:,1]

        y = np.maximum(y, 0) 

        y = np.sqrt(y)

        y = savitzkyGolay(y, window_length=11)

        z = baseline_als(y, 10**9, 0.001)
        y = np.maximum(0, y-z).astype(np.float)
        # plt.plot(x, y ,color='y')
        # plt.show()


        bList = []
        bList_y = []
        interval = 0.5
        
        if sType == 'S2':
            aList = [103.9136, 108.90324, 164.9097]
        elif sType == 'C2':
            aList = [146.01622, 265.91915, 656.02131]
        elif sType in ['A6','A8']:
            aList = [196.9738, 393.9404, 590.907]
        else:
            aList = []

        for a in aList:
            xSubList = np.logical_and(a - interval <= x, x <= a + interval)
            ySubList = y[xSubList]

            firstIndex = np.where(xSubList==True)[0][0]
            subIndex = np.argmax(ySubList)
            if subIndex == 0 or subIndex == (len(ySubList) - 1):
                print(cnt, totalNum)
                print(path)
            totalIndex = firstIndex + subIndex
            bList.append(x[totalIndex])

        if len(bList) != 0:
            x_new = []
            for x_item in x:
                new = x_item
                if x_item < bList[0] or x_item >= bList[2]:
                    new = align(aList[0], aList[2], bList[0], bList[2], x_item)
                elif x_item >= bList[0] and x_item < bList[1]:
                    new = align(aList[0], aList[1], bList[0], bList[1], x_item)
                elif x_item >= bList[1] and x_item < bList[2]:
                    new = align(aList[1], aList[2], bList[1], bList[2], x_item)
                x_new.append(new)
            df['mass'] = x_new
        else:
            df['mass'] = x

        df['intensity'] = y
        df.to_csv(os.path.join(afterpath, path), index=False)

if __name__ == "__main__":
    # The path where the CSV format file is located
    DATA_PATH='/opt/BioData_Base/data20190708/csv'
    # The path where the processed file will be saved
    DATA_PREPROCESS_PATH='/opt/BioData_Base/data20190708/csv-after-py'
    
    main(2,DATA_PATH, DATA_PREPROCESS_PATH)