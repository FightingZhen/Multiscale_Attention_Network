import matplotlib.pyplot as plt
import numpy as np

scorefile_dir = 'D:/Workspace/DLD_Classification/model_results/score/'
lablefile_dir = 'D:/Workspace/DLD_Classification/model_results/label/'


def readColumn(fileName, target_column):
    score_list = []
    count_array = np.zeros([101])
    with open(scorefile_dir + fileName, 'r') as f:
        for line in f:
            l = line[:-2].split(' ')
            l = list(map(float, l))
            score_list.append(l[target_column])
    for k in range(len(score_list)):
        score_list[k] = int(score_list[k] * 100)

    for m in range(len(score_list)):
        count_array[score_list[m]] += 1

    print(count_array)
    # print(score_list)
    return count_array


def plotScores(scoreList):
    plt.figure()
    plt.plot(np.arange(0, 101), scoreList)
    plt.show()


if __name__ == '__main__':
    count_array = readColumn('db_resnet_y_score.txt', target_column=1)
    plotScores(count_array)
