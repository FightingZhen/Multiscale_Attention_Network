import numpy as np

source_file = 'D:/tempOperationArea/probs.txt'
out_score_file = 'D:/uchiyama_y_score.txt'
out_label_file = 'D:/uchiyama_test_label.txt'


def generateOneHotVector(number):
    vec = np.zeros([7, ])
    vec[number] = 1

    return vec


def extractInformation(src_file):
    f = open(src_file, 'r')
    for line in f:
        l = line[:-2].split(' ')

        lab = generateOneHotVector(int(l[0]) - 1)
        lab = list(map(str, lab))
        lab = ' '.join(lab)
        sco = str(' '.join(l[1:]))

        with open(out_score_file, 'a+') as f_s:
            f_s.write(sco + ' \n')
        with open(out_label_file, 'a+') as f_l:
            f_l.write(lab + ' \n')
    f.close()


if __name__ == '__main__':
    extractInformation(source_file)
