import numpy as np
import os
import re


def getFileName(fdir):
    return os.listdir(fdir)


def getRootFileName(labelFileDir):
    flist = getFileName(labelFileDir)
    root = []
    for f in flist:
        root.append(re.sub('_test_label.txt', '', f))

    return root


def str2matrix(filename, type):
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            l = list(line[:-2].split(' '))
            for i in range(len(l)):
                l[i] = float(l[i])
            matrix.append(l)

    np_matrix = np.array(matrix, dtype=type)

    return np_matrix


def checkDir(tar_dir):
    if not os.path.isdir(tar_dir):
        os.makedirs(tar_dir)


def compareArrays(arr1, arr2):
    ind1 = np.argmax(arr1)
    ind2 = np.argmax(arr2)
    if ind1 != ind2:
        return arr1, arr2
    else:
        return [0], [0]


def save2file(filename, filedir, content, mode='a+'):
    checkDir(filedir)
    with open(filedir + filename, mode) as f:
        f.write(content + '\n')


def arrs2str(arr):
    str_arr = list(map(str, arr))

    return ' '.join(str_arr)


def findMisClassifiedSamples(label, score):
    mat_height, mat_width = label.shape

    for i in range(mat_height):
        lab, sc = compareArrays(score[i], label[i])
        if len(lab) > 1:
            save2file('misclassified.txt', save_dir, str(i) + ' ' + arrs2str(lab) + ' ' + arrs2str(sc) + '\n')


if __name__ == '__main__':
    label_fdir = 'D:/Workspace/DLD_Classification/model_results_ROC/after-softmax/label/'
    score_fdir = 'D:/Workspace/DLD_Classification/model_results_ROC/after-softmax/score/'

    save_dir = 'D:/Workspace/DLD_Classification/model_results_ROC/after-softmax/misclassified/'

    rootFileName = getRootFileName(label_fdir)
    print(rootFileName)


    scoreFileName = 'fc_attention_y_score.txt'
    labelFileName = 'fc_attention_test_label.txt'

    lab_mat = str2matrix(label_fdir + labelFileName, type=np.int32)
    sc_mat = str2matrix(score_fdir + scoreFileName, type=np.float32)

    findMisClassifiedSamples(lab_mat, sc_mat)
