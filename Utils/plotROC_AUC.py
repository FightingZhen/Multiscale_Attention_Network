from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interp
from PIL import Image


def str2matrix(filename, type):
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            l = list(line[:-2].split(' '))
            for i in range(len(l)):
                if type == 'float':
                    l[i] = float(l[i])
                if type == 'int':
                    l[i] = int(float(l[i]))
                # print(l[i])
            matrix.append(l)

    np_matrix = np.array(matrix, dtype=type)

    return np_matrix


def gatherMatrices(labeldir, scoredir):
    label_filelist = getFileList(labeldir)
    score_filelist = getFileList(scoredir)
    print(label_filelist)
    print(score_filelist)

    label_lib = []
    score_lib = []

    for l, s in zip(label_filelist, score_filelist):
        mat_l = str2matrix(label_fdir + l, type='int')
        mat_s = str2matrix(score_fdir + s, type='float')
        label_lib.append(mat_l)
        score_lib.append(mat_s)

    # print(len(label_lib))
    # print(len(score_lib))
    # print(label_lib[0].shape)
    # print(score_lib[0].shape)

    # print(label_lib)
    # print(score_lib)

    return label_lib, score_lib


# def getRootFileName(labelFileDir):
#     flist = getFileList(labelFileDir)
#     root = []
#     for f in flist:
#         root.append(re.sub('_test_label.txt', '', f))
#
#     # print(root)
#     return root


def getFileList(fileDir):
    return os.listdir(fileDir)


def plotROC(labelArray, scoreArray, mode='after-softmax'):
    pattern_name = {0: 'Consolidation',
                    1: 'Multi-focal Ground-glass Opacity',
                    2: 'Honey Combining',
                    3: 'Reticular Ground-glass Opacity',
                    4: 'Emphysema',
                    5: 'Nodular Opacity',
                    6: 'Normal Pulmonary Textures'}

    model_name = {0: 'BOF',
                  1: 'DB-Resnet',
                  2: 'Multi-scale Residual Network',
                  3: 'LeNet',
                  4: 'ResNet-50',
                  5: 'TMI',
                  6: 'SpeDesFea',
                  7: 'VGG'}

    num_class = len(pattern_name)
    num_model = len(model_name)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    saveDir = '../ROC_pictures/' + mode + '/'
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)

    for i in range(num_class):
        fpr[i] = dict()
        tpr[i] = dict()
        roc_auc[i] = dict()
        for j in range(num_model):
            fpr[i][j], tpr[i][j], _ = roc_curve(labelArray[j][:, i], scoreArray[j][:, i])
            # print('class ' + pattern_name[i] + ' model ' + str(j) + str(_.shape))
            roc_auc[i][j] = auc(fpr[i][j], tpr[i][j])

        plt.figure(figsize=[8, 6])
        lw = 2
        plt.plot(fpr[i][0], tpr[i][0], color='yellow',
                 lw=lw, label=model_name[0] + ' roc curve (auc = %0.4f)' % roc_auc[i][0])
        plt.plot(fpr[i][1], tpr[i][1], color='red',
                 lw=lw, label=model_name[1] + ' roc curve (auc = %0.4f)' % roc_auc[i][1])
        plt.plot(fpr[i][2], tpr[i][2], color='blue',
                 lw=lw, label=model_name[2] + ' roc curve (auc = %0.4f)' % roc_auc[i][2])
        plt.plot(fpr[i][3], tpr[i][3], color='pink',
                 lw=lw, label=model_name[3] + ' roc curve (auc = %0.4f)' % roc_auc[i][3])
        plt.plot(fpr[i][4], tpr[i][4], color='orange',
                 lw=lw, label=model_name[4] + ' roc curve (auc = %0.4f)' % roc_auc[i][4])
        plt.plot(fpr[i][5], tpr[i][5], color='skyblue',
                 lw=lw, label=model_name[5] + ' roc curve (auc = %0.4f)' % roc_auc[i][5])
        plt.plot(fpr[i][6], tpr[i][6], color='teal',
                 lw=lw, label=model_name[6] + ' roc curve (auc = %0.4f)' % roc_auc[i][6])
        plt.plot(fpr[i][7], tpr[i][7], color='cyan',
                 lw=lw, label=model_name[7] + ' roc curve (auc = %0.4f)' % roc_auc[i][7])

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(pattern_name[i])
        plt.legend(loc="lower right")
        plt.savefig(saveDir + pattern_name[i] + '.png')

    return tpr, fpr, roc_auc


def plotMeanROC(tpr, fpr, roc_auc, mode='after-softmax', num_class=7):
    color = ['yellow', 'red', 'blue', 'pink', 'orange', 'skyblue', 'teal', 'cyan']

    model_name = {0: 'BOF',
                  1: 'DB-Resnet',
                  2: 'Multi-scale Residual Network',
                  3: 'LeNet',
                  4: 'ResNet-50',
                  5: 'TMI',
                  6: 'SpeDesFea',
                  7: 'VGG'}

    num_model = len(model_name)

    tpr["macro"] = dict()
    fpr["macro"] = dict()
    roc_auc["macro"] = dict()

    saveDir = '../ROC_pictures/' + mode + '/'
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)

    for i in range(num_model):
        all_fpr = np.unique(np.concatenate([fpr[j][i] for j in range(num_class)]))
        mean_tpr = np.zeros_like(all_fpr)
        for k in range(num_class):
            mean_tpr += interp(all_fpr, fpr[k][i], tpr[k][i])

        mean_tpr /= num_class

        fpr["macro"][i] = all_fpr
        tpr["macro"][i] = mean_tpr
        roc_auc["macro"][i] = auc(fpr["macro"][i], tpr["macro"][i])

    plt.figure(figsize=[8, 6])
    lw = 2
    for i in range(num_model):
        plt.plot(fpr["macro"][i], tpr["macro"][i], color=color[i],
                 lw=lw, label=model_name[i] + ' roc curve (area = %0.4f)' % roc_auc["macro"][i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average')
    plt.legend(loc="lower right")
    plt.savefig(saveDir + 'Mean.png')


def fusePlots(subPlotDir, mode='after-softmax'):
    subPlotFList = getFileList(subPlotDir + mode + '/')
    print(subPlotFList)
    images = [Image.open(subPlotDir + mode + '/' + subPlotFList[i]) for i in range(len(subPlotFList))]

    width, height = images[0].size

    saveDir = '../ROC_pictures/' + mode + '/fusedImage.png'

    fuseImg = Image.new('RGB', (width * 2, height * 4))
    fuseImg.paste(images[0], box=(0, 0))
    fuseImg.paste(images[4], box=(1 * width, 0))
    fuseImg.paste(images[2], box=(0, 1 * height))
    fuseImg.paste(images[7], box=(1 * width, 1 * height))
    fuseImg.paste(images[1], box=(0, 2 * height))
    fuseImg.paste(images[5], box=(1 * width, 2 * height))
    fuseImg.paste(images[6], box=(0, 3 * height))
    fuseImg.paste(images[3], box=(1 * width, 3 * height))

    fuseImg.save(saveDir)


if __name__ == '__main__':
    mode = 'after-softmax'
    label_fdir = 'D:/Workspace/DLD_Classification/model_results_ROC/' + mode + '/label/'
    score_fdir = 'D:/Workspace/DLD_Classification/model_results_ROC/' + mode + '/score/'

    lab_lib, sc_lib = gatherMatrices(label_fdir, score_fdir)
    tpr, fpr, roc_auc = plotROC(lab_lib, sc_lib, mode=mode)
    plotMeanROC(tpr, fpr, roc_auc, mode)
    fusePlots('../ROC_pictures/', mode=mode)
