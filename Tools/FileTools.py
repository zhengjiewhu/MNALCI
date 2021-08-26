import os, shutil
def makeDir(path):
    path=path.strip()
    path=path.rstrip("\\")

    isExists = os.path.exists(path)
    if isExists:
        pass
    else:
        os.makedirs(path)

def writeRes(identify, res, path):
    with open(path, 'a') as f:
        f.write(identify + '\n')
        f.write(res + '\n')
        f.write('--------------------------------' + '\n')

def exportFileList(fileListList, path):
    with open(path, 'w+') as f:
        if len(fileListList) == 2:
            for i,j in zip(fileListList[0], fileListList[1]):
                f.write(i+'\t'+j)
                f.write('\n')
        else:
            for i in fileListList[0]:
                f.write(i)
                f.write('\n')

def copyFiles(ori_dir, new_dir):
    for root, dirs, files in os.walk(ori_dir):
        for file in files:
            src_file = os.path.join(root, file)
            shutil.copy(src_file, new_dir)

def copyFile(src_file, new_file):
    shutil.copy(src_file, new_file)

if __name__ == "__main__":
    ori_dir = '/opt/BioData_Sample/data20190522_0614/csv'
    new_dir = '/opt/BioData/new/csv'

    copyFiles(ori_dir, new_dir)