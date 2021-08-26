import os
import pathlib
import shutil
import Tools.FileTools as FileTools
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import csv

# Define the root directory and the path prefix
ROOT_DIR = '/opt/BioData'
SUB_DIR = 'paper'

BASE_DIR = os.path.join(ROOT_DIR, SUB_DIR)
mzmlDir = os.path.join(BASE_DIR, 'mzml')
rmzmlDir = os.path.join(BASE_DIR, 'rmzml')
csvDir = os.path.join(BASE_DIR, 'csv')
csvAfterDir = os.path.join(BASE_DIR, 'csv-after')

def makeDirs():
    dirList = ['rmzml','mzml','csv','csv-after','csv-after-py','csv-after-r','tempData', 'resData', 'csv-peak-absi', 'csv-peak-area']
    for i in dirList:
        FileTools.makeDir(os.path.join(ROOT_DIR, SUB_DIR, i))

def convertToCsv():
    importr('MALDIquantForeign')
    rCode = '''
        mzmlDir <- '%s'
        first_category_name = list.files(file.path(mzmlDir))
        n = length(first_category_name)
        for(i in 1:n){
        my_spectra <- importMzMl(file.path(mzmlDir, first_category_name[i]),
                                verbose=FALSE,
                                # massRange=c(100,1000),
                                # excludePattern="/TC|/2/1SRef"
                                excludePattern="/2/1SRef"
                                )
        print(paste(i,n,sep='/'))
        fileName = paste(first_category_name[i], '.csv',sep='')
        exportCsv(my_spectra,path='%s',file=file.path('%s',fileName),force=TRUE)
        }
        ''' % (mzmlDir, csvDir, csvDir)
    print(rCode)
    robjects.r(rCode)


def list_all_files(rootPath, level):
    _files = []
    listDir = os.listdir(rootPath)
    if level == 0:
        for i in range(0,len(listDir)):
            path = os.path.join(rootPath,listDir[i])
            if os.path.isdir(path):
                _files.append(path)
    elif level == 1:
        for i in range(0,len(listDir)):
            path = os.path.join(rootPath,listDir[i])
            if os.path.isdir(path):
                subListDir = os.listdir(path)
                for j in range(0, len(subListDir)):
                    subpath = os.path.join(path,subListDir[j])
                    if os.path.isdir(subpath):
                        _files.append(subpath)
    elif level == 2:
        for i in range(0,len(listDir)):
            path = os.path.join(rootPath,listDir[i])
            if os.path.isdir(path):
                subListDir = os.listdir(path)
                for j in range(0, len(subListDir)):
                    subpath = os.path.join(path,subListDir[j])
                    if os.path.isdir(subpath):
                        subSubListDir = os.listdir(subpath)
                        for k in range(0, len(subSubListDir)):
                            subSubPath = os.path.join(subpath,subSubListDir[k])
                            # if os.path.isdir(subSubPath):
                            _files.append(subSubPath)

    return _files

def extractMZML(rootPath, toPath, level):
    for dateDir in os.listdir(rootPath):
        dirList = list_all_files(os.path.join(rootPath, dateDir), level)
        for ori_dir in dirList:
            dirName = ori_dir.split('/')[-1] 
            if level == 0:
                fileContainDir = ''
            elif level == 1:
                fileContainDir = ori_dir.split('/')[-2]
            elif level == 2:
                fileContainDir = ori_dir.split('/')[-2] + '_' + ori_dir.split('/')[-3]
            # Rename file
            if os.path.isdir(ori_dir):
                new_dir = os.path.join(toPath, dirName + '_' + fileContainDir)
                new_dir = new_dir.replace('.','X')
            else:
                new_dir = os.path.join(toPath, 
                        dirName.split('.')[0] + '_' + fileContainDir + '.' + dirName.split('.')[1])
            if os.path.isdir(ori_dir):
                shutil.copytree(ori_dir, new_dir)
            else:
                shutil.copyfile(ori_dir, new_dir)

        for item in os.listdir(rootPath):
            subpath = os.path.join(rootPath,item)
            if len(os.listdir(subpath)) == 0:
                os.rmdir(subpath)

        print('Extract SUCCESS')

if __name__ == "__main__":
    makeDirs()
    extractMZML(rmzmlDir, mzmlDir, 2)
    convertToCsv()
