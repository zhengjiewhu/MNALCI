import sys
sys.path.append('../')
import tempfile
from sklearn.datasets import load_files
import numpy as np
import pandas as pd
import csv
import os, shutil
import Tools.FileTools as FileTools
import bioCode.config as config

Interval = config.Interval
MIN_MZ = config.MIN_MZ
MAX_MZ = config.MAX_MZ
def load_data(dataFoldPath, shuffle=True):
    dataBunch = load_files(container_path=dataFoldPath,
                      encoding="utf-8", shuffle=shuffle)

    labels = np.array(dataBunch.target)

    n_samples = len(dataBunch.data)
    n_features = int((MAX_MZ - MIN_MZ) / Interval)

    dataG = np.zeros((n_samples, n_features))
    keys = np.zeros((n_samples, n_features))

    for i in range(len(dataBunch.data)):
        lines = dataBunch.data[i].split('\n')[1:-1]
        for line in lines:
            mass = float(line.split(',')[0])
            if mass < MIN_MZ or mass >= MAX_MZ:
                continue
            value = float(line.split(',')[1])
            # value = int(float(line.split(',')[1]))
            grid = int((mass - MIN_MZ) / Interval)
            dataG[i, grid] += value

    return dataG, labels, dataBunch.target_names, keys, dataBunch.filenames

def combCsv(posList, negList, dataPath, resPath):
    tempDir = tempfile.mkdtemp()
    posDir = os.path.join(tempDir, '1')
    negDir = os.path.join(tempDir, '0')
    FileTools.makeDir(posDir)
    FileTools.makeDir(negDir)
    baseDir = dataPath
    for item in posList:
        oriPath = os.path.join(baseDir, item)
        (_, filename) = os.path.split(item)
        dstPath = os.path.join(posDir, filename)
        shutil.copyfile(oriPath, dstPath)
    for item in negList:
        oriPath = os.path.join(baseDir, item)
        (_, filename) = os.path.split(item)
        dstPath = os.path.join(negDir, filename)
        shutil.copyfile(oriPath, dstPath)
    
    features, labels, target_names , _, filenames= load_data(tempDir, False)
    print('Loading...'+tempDir)
    headLine = ['PrimaryID', 'category'] + [MIN_MZ+i*Interval for i in range(int((MAX_MZ-MIN_MZ)/Interval))]
    num_data, _ = features.shape
    file_list = []
    with open(os.path.join(resPath), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headLine)
        for i in range(num_data):
            _, fname = os.path.split(filenames[i])
            file_list.append(fname)
            line = [fname,  target_names[labels[i]]] + list(features[i])
            writer.writerow(line)
    print('Creating temp file...' + resPath)
    return file_list


def judgeRepeated(array):
    nums={}
    for i in array:
        if i not in nums:
            nums[i]=True
        else:
            return True
    return False

def combMultiCsv(fileListList, tagList, dataPath, resPath):
    assert(len(fileListList)==len(tagList))
    assert(judgeRepeated(tagList)==False)
    tempDir = tempfile.mkdtemp()
    tagDirList = []
    for i in tagList:
        tagDir = os.path.join(tempDir, str(i))
        tagDirList.append(tagDir)
        FileTools.makeDir(tagDir)
    baseDir = dataPath
    for index in range(len(tagDirList)):
        for item in fileListList[index]:
            oriPath = os.path.join(baseDir, item)
            (_, filename) = os.path.split(item)
            dstPath = os.path.join(tagDirList[index], filename)
            shutil.copyfile(oriPath, dstPath)
   
    features, labels, target_names , _, filenames= load_data(tempDir, False)
    print('Load file complete:'+tempDir)
    headLine = ['PrimaryID', 'category'] + [MIN_MZ+i*Interval for i in range(int((MAX_MZ-MIN_MZ)/Interval))]
    num_data, _ = features.shape
    file_list = []
    with open(os.path.join(resPath), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headLine)
        for i in range(num_data):
            _, fname = os.path.split(filenames[i])
            file_list.append(fname)
            line = [fname,  target_names[labels[i]]] + list(features[i])
            writer.writerow(line)
    print('Create temp file complete:' + resPath)
    return target_names, file_list

if __name__ == "__main__":
    pos_list = ['BRDS_4M0185103_A6_SHZL_001_20190227_D07.csv']
    neg_list = ['PPHP_4M0185103_A6_SHZS_1002227_20190226_E03.csv']
    combCsv(pos_list, neg_list, '/opt/RData/totalCsvOut', '/opt/RData/res/1.csv')