import sys
sys.path.append("./")
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn import preprocessing
import Tools.FileTools as FileTools
import os
import math
import random
import copy
import functools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import Tools.DataTools as DataTools
from functools import reduce
import config
import time


def getXY(posFileList, negFileList, tempFileName, dataPath, resBasePath, returnList=False, checkFileName=True):
    assert(len(posFileList) == len(negFileList))
    x_list = []
    y_list = []
    filenames_list = []
    for i in range(len(posFileList)):
        filePath = os.path.join(resBasePath, tempFileName+'-'+str(i)+'.csv')
        file_list = DataTools.combCsv(posFileList[i], negFileList[i], dataPath, filePath)

        # rawDataT = np.loadtxt(filePath, delimiter=',', dtype=np.str)[1:, :]
        # print(rawDataT.shape)
        # print(rawDataT[0])
        rawDataT = pd.read_csv(filePath, delimiter=',', header=None).values[1:, :]
        # print(rawDataT.shape)
        # print(rawDataT[0])
        
        x = np.float64(rawDataT[:, 2:])
        y = np.int32(rawDataT[:, 1])

        # x = preprocessing.scale(x)

        # x = signal.medfilt(x, (1,5))


        # np.savetxt(config.RES_BASE_PATH + '/%s-x1.csv' % tempFileName, np.hstack((rawDataT[:, 0].reshape(x.shape[0],1), x)) , fmt='%s' , delimiter=',')
        x = x / np.sum(x, axis=1)[:,None] * 10000 
        # x = [signal_decomp(ix) for ix in x]
        # x = np.array(x)
        # print(x.shape)                
        # np.savetxt(config.RES_BASE_PATH + '/%s-x2.csv' % tempFileName, np.hstack((rawDataT[:, 0].reshape(x.shape[0],1), x)) ,fmt='%s', delimiter=',')
        

        x_list.append(x)
        y_list.append(y)
        filenames_list.append(file_list)

    for i in range(0, len(y_list)):
        for j in range(i+1, len(y_list)):
            assert((y_list[i] == y_list[j]).all())


    if checkFileName:
        patientID_list = []
        for file_list in filenames_list:
            patientID_list.append(
                [f.split('_')[0] + f.split('_')[3] +f.split('_')[4] for f in file_list]
                )
        for i in range(0, len(patientID_list)):
            for j in range(i+1, len(patientID_list)):
                assert((np.array(patientID_list[i]) == np.array(patientID_list[j])).all())

    if returnList:
        return x_list, y_list, filenames_list

    X = np.hstack(x_list)
    Y = y_list[0]
    files = filenames_list[0]
    return X, Y, files

def getXY_Multi(fileListList, tagList, tempFileName, dataPath, resBasePath, returnList=False, checkFileName=True):
    for i in range(0, len(fileListList)):
        for j in range(i+1, len(fileListList)):
            class1List = fileListList[i]
            class2List = fileListList[j]
            assert(len(class1List) == len(class2List))

    x_list = []
    y_list = []
    filenames_list = []
    for sTypeIndex in range(len(fileListList[0])):
        tempFileListList = []
        for classIndex in range(len(fileListList)):
            tempFileListList.append(fileListList[classIndex][sTypeIndex])

        fileOutputPath = os.path.join(
            resBasePath, tempFileName+'-'+str(sTypeIndex)+'.csv')
        target_names, file_list = DataTools.combMultiCsv(
            tempFileListList, tagList, dataPath, fileOutputPath)

        # rawDataT = np.loadtxt(
        #     fileOutputPath, delimiter=',', dtype=np.str)[1:, :]
        rawDataT = pd.read_csv(fileOutputPath, delimiter=',', header=None).values[1:, :]
        x = np.float64(rawDataT[:, 2:])
        y = rawDataT[:, 1]
        
        # x = preprocessing.scale(x)
        # x = signal.medfilt(x, (1,5))


        # TIC,
        x = x / np.sum(x, axis=1)[:,None] * 10000

        x_list.append(x)
        y_list.append(y)
        filenames_list.append(file_list)

    X = np.hstack(x_list)

    for i in range(0, len(y_list)):
        for j in range(i+1, len(y_list)):
            assert((y_list[i] == y_list[j]).all())
    Y = y_list[0]

    if checkFileName:
        patientID_list = []
        for file_list in filenames_list:
            patientID_list.append(
                [f.split('_')[0] + f.split('_')[3] +f.split('_')[4] for f in file_list]
                )
        for i in range(0, len(patientID_list)):
            for j in range(i+1, len(patientID_list)):
                assert((np.array(patientID_list[i]) == np.array(patientID_list[j])).all())

    if returnList:
        return x_list, y_list, target_names, filenames_list
    else:
        return X, Y, target_names, filenames_list[0]

def oneStepCV_Vote_Final_Train(X_list_train, y_list_train, batchID, modelID, multiClass=False, 
    threshold=None, para_List=None):
    model_cnt = len(X_list_train)
    model_list = []
    typeDM = 'M' if multiClass else 'D'
    if threshold == None:
        threshold = model_cnt / 2

    if para_List == None:
        para_List = [{} for i in range(model_cnt)]

    FileTools.makeDir('finalModels/' + batchID)
    with open(os.path.join('finalModels/', batchID, typeDM+'-mNames.txt'), 'w+') as f:
        for i in range(model_cnt):
            model_list.append(
                SVC(kernel='linear', probability=True, class_weight='balanced', **para_List[i])
                # CalibratedClassifierCV(LinearSVC(max_iter=10000))  
                )
            model_list[i].fit(X_list_train[i], y_list_train[i])
            mName = '-'.join([typeDM, str(i), modelID, 'T' + str(threshold).replace('.', '_'),
                              'P'+str(para_List[i])])+'.m'
            mPath = os.path.join('finalModels/', batchID, mName)

            f.write(mPath + '\n')
            joblib.dump(model_list[i], mPath)

def start_Vote_AllData(posFileList, negFileList, threshold=None):
    if len(posFileList[0]) == 0 or len(negFileList[0]) == 0:
        raise Exception("No data")
    model_cnt = len(posFileList)
    if threshold == None:
        threshold = model_cnt / 2

    mID = "test"
    # mPath = os.path.join('models/', mID+'.m')

    X_list, y_list, _ = getXY(posFileList, negFileList, mID+'-TRAIN',
                           config.ROOT_DIR, config.RES_BASE_PATH, True)
    
    oneStepCV_Vote_Final_Train(X_list, y_list, config.MODEL_ID, mID, False,
                               threshold, None)

def start_Vote_MultiClass_AllData(fileListList, tagList):
    uniID = getUniqueId_Multi(tagList)
    # fileListList = []
    # for classDict in classDictList:
    #     classFileList = MysqlORM.selectItems(**classDict)
    #     if len(classFileList[0]) == 0:
    #         raise Exception(str(classDict)+"data not found")
    #     fileListList.append(classFileList)

    x_list, y_list, target_names, _ = getXY_Multi(fileListList, tagList, uniID+'-TRAIN',
                                               config.ROOT_DIR, config.RES_BASE_PATH, True)
    oneStepCV_Vote_Final_Train(x_list, y_list, config.MODEL_ID, uniID, True)
    
    with open(os.path.join('finalModels/', config.MODEL_ID, 'target_names.txt'), 'w+') as f:
        for i in range(len(target_names)):
            f.write(str(target_names[i]) + '\n')

if __name__ == "__main__":
    # Train model
    posFileList = []
    negFileList = []
    threshold = 0.5
    for i in os.listdir("data/csv/pos"):
        posFileList.append(os.path.join("data/csv/pos", i))
    for i in os.listdir("data/csv/neg"):
        negFileList.append(os.path.join("data/csv/neg", i))
    start_Vote_AllData(posFileList, negFileList, threshold)
    
    # Multi class
    # start_Vote_MultiClass_AllData(disease_fileListList, tagList)
