import matplotlib.pyplot as plt

plt.switch_backend('agg')
import pickle
import os
import numpy as np

train_val_los_path = 'D:/TMI_Data/DLD_pickle_data/'


def loadPickle(pklFilePath, pklFileName):
    with open(pklFilePath + pklFileName, 'rb') as f:
        message = pickle.load(f)

    return message


def getFileNameList(filePath):
    l = os.listdir(filePath)
    l = sorted(l, key=lambda x: x[:x.find('.')])

    return l


def cal_mean_std(filePath, mode):
    img_fileName = mode + '_image.pkl'
    lab_fileName = mode + '_label.pkl'

    img_array = loadPickle(filePath, img_fileName)
    lab_array = loadPickle(filePath, lab_fileName)

    print('Mode: ', mode)

    splitIntoClasses(img_array, lab_array)


def splitIntoClasses(imageArray, labelArray):
    class_lib = {0: 'CON',
                 1: 'M-GGO',
                 2: 'HCM',
                 3: 'R-GGO',
                 4: 'EMP',
                 5: 'NOD',
                 6: 'NOR'}

    img_class = {0: [],
                 1: [],
                 2: [],
                 3: [],
                 4: [],
                 5: [],
                 6: []}

    num = imageArray.shape[0]

    for i in range(num):
        lab = list(labelArray[i]).index(labelArray.max())
        # print(lab)
        img_class[lab].append(imageArray[i])

    for j in range(7):
        _img = np.array(img_class[j])
        print('Category ' + class_lib[j] + ' mean: %g, std: %g' % (_img.mean(), _img.std()))


def readLogFile(filePath, fileName):
    training_acc = []
    validation_acc = []
    test_acc = []

    with open(filePath + fileName, 'r') as f:
        lines = f.readlines()

        for i in range(len(lines)):
            l = lines[i].lower()
            if l.find('epoch') == 0 and l.find('validation') > -1:
                tra_start = l.index('training')
                tra_end = l.index(' ,validation')
                tra = float(l[tra_start:tra_end].split('[')[1][:-1])
                training_acc.append(tra)

                val_start = l.index('validation')
                val_end = l.index(' , loss')
                val = float(l[val_start:val_end].split('[')[1][:-1])
                validation_acc.append(val)

            if l.find('test accuracy') == 0:
                tst = float(l.split(':')[1][1:])
                test_acc.append(tst)

    # print(training_acc)
    # print(validation_acc)
    # print(test_acc)
    # print(len(training_acc))
    # print(len(validation_acc))
    # print(len(test_acc))
    return training_acc, validation_acc, test_acc


def plot_acc_loss(training_acc, validation_acc, test_acc):
    plt.figure(figsize=(20.48, 10.24))
    plt.plot(range(200), training_acc, linewidth=1.0, linestyle='-', label='training_accuracy')
    plt.plot(range(200), validation_acc, linewidth=1.0, color='red', linestyle='--',
             label='validation_accuracy')
    plt.plot(range(200), test_acc, linewidth=1.0, color='grey', linestyle='--',
             label='test_accuracy')
    plt.title('Accuracy')
    plt.ylim([0.0, 1.2])
    plt.xticks([x for x in range(0, 200 + 1, 10)])
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(train_val_los_path + 'Accuracy.png')


if __name__ == '__main__':
    # cal_mean_std(train_val_los_path, 'training')
    # cal_mean_std(train_val_los_path, 'validation')
    # cal_mean_std(train_val_los_path, 'test')

    train_acc, validation_acc, test_acc = readLogFile(train_val_los_path, 'resnet_multifc_fc_attention.txt')
    plot_acc_loss(train_acc, validation_acc, test_acc)

