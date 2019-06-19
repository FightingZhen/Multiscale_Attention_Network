import numpy as np

score_dir = 'D:/Workspace/DLD_Classification/Major_Review/McNeMar_Test/score/'
label_dir = 'D:/Workspace/DLD_Classification/Major_Review/McNeMar_Test/label/'


def locateMaximumPrediction(line):
    return np.where(line == np.max(line))[0][0]


def extractMatrixFromTxt(txtFileName, txtFileDir):
    extractedMat = []
    with open(txtFileDir + txtFileName, 'r') as f:
        for line in f:
            l = line[:-2].split(' ')
            l = list(map(float, l))
            extractedMat.append(l)

    extractedMat_np = np.array(extractedMat, dtype=float)

    return extractedMat_np


def generateOneVsAllMatrix(probabilityMatrix):
    mat_height, mat_width = probabilityMatrix.shape

    oneVsAllMat = np.zeros([mat_height, mat_width])

    for i in range(mat_height):
        oneVsAllMat[i][locateMaximumPrediction(probabilityMatrix[i])] = 1

    return oneVsAllMat


def generateClassificationResponse(oneVsAllMatrix, lableMatrix, category):
    proposed_mat = oneVsAllMatrix[:, category]
    compared_mat = lableMatrix[:, category]

    mcnemar_mat = []
    for i in range(proposed_mat.shape[0]):
        if proposed_mat[i] == compared_mat[i] == 1:
            mcnemar_mat.append(0)
        else:
            mcnemar_mat.append(1)

    mcnemar_mat = np.array(mcnemar_mat, dtype=int)

    return mcnemar_mat


def generateMcNeMarMatrix(method1, method2):
    mcnemar_mat = np.zeros([2, 2])
    for i in range(method1.shape[0]):
        mcnemar_mat[method1[i]][method2[i]] += 1

    return mcnemar_mat


def calculateMcnemarMatrix(proposedScoreFile, proposedLableFile, comparableScoreFile, comparableLableFile, category):
    psf_mat = extractMatrixFromTxt(proposedScoreFile, score_dir)
    plf_mat = extractMatrixFromTxt(proposedLableFile, label_dir)
    csf_mat = extractMatrixFromTxt(comparableScoreFile, score_dir)
    clf_mat = extractMatrixFromTxt(comparableLableFile, label_dir)

    psf_mat_one_vs_all = generateOneVsAllMatrix(psf_mat)
    csf_mat_one_vs_all = generateOneVsAllMatrix(csf_mat)

    proposed_classification_res = generateClassificationResponse(psf_mat_one_vs_all, plf_mat, category)
    compared_classification_res = generateClassificationResponse(csf_mat_one_vs_all, clf_mat, category)

    mcnemar_mat = generateMcNeMarMatrix(proposed_classification_res, compared_classification_res)

    return mcnemar_mat


def calculateMcnemarMeanMatrix(proposedScoreFile, proposedLableFile, comparableScoreFile, comparableLableFile):
    psf_mat = extractMatrixFromTxt(proposedScoreFile, score_dir)
    plf_mat = extractMatrixFromTxt(proposedLableFile, label_dir)
    csf_mat = extractMatrixFromTxt(comparableScoreFile, score_dir)
    clf_mat = extractMatrixFromTxt(comparableLableFile, label_dir)

    psf_mat_one_vs_all = generateOneVsAllMatrix(psf_mat)
    csf_mat_one_vs_all = generateOneVsAllMatrix(csf_mat)

    total_proposed_classification_res = []
    total_compared_classification_res = []

    for i in range(7):
        proposed_classification_res = generateClassificationResponse(psf_mat_one_vs_all, plf_mat, i)
        compared_classification_res = generateClassificationResponse(csf_mat_one_vs_all, clf_mat, i)

        for item_p, item_c in zip(proposed_classification_res, compared_classification_res):
            total_proposed_classification_res.append(item_p)
            total_compared_classification_res.append(item_c)

    total_proposed_classification_res = np.array(total_proposed_classification_res)
    total_compared_classification_res = np.array(total_compared_classification_res)

    mcnemar_mean_mat = generateMcNeMarMatrix(total_proposed_classification_res, total_compared_classification_res)

    return mcnemar_mean_mat


def computeVariance(mcnemarMatrix):
    return (mcnemarMatrix[0][1] - mcnemarMatrix[1][0]) ** 2 / (mcnemarMatrix[0][1] + mcnemarMatrix[1][0])


def main():
    proposedMethod = 'fc_attention'
    comparedMethod = ['db_resnet', 'lenet', 'tmi', 'BOF', 'vgg', 'resnet_50', 'uchi']

    proposedMethod_ScoreFile = proposedMethod + '_y_score.txt'
    proposedMethod_LableFile = proposedMethod + '_test_label.txt'

    comparedMethod_ScoreFile = [x + '_y_score.txt' for x in comparedMethod]
    comparedMethod_LableFile = [y + '_test_label.txt' for y in comparedMethod]

    # print(comparedMethod_ScoreFile)
    # print(comparedMethod_LableFile)

    for i in range(len(comparedMethod_LableFile)):
        for j in range(7):
            tmp_mcnemarMatrix = calculateMcnemarMatrix(proposedMethod_ScoreFile, proposedMethod_LableFile,
                                                       comparedMethod_ScoreFile[i], comparedMethod_LableFile[i], j)
            print(tmp_mcnemarMatrix.reshape([4, ]).tolist())
            # print(tmp_mcnemarMatrix)
            print('Category %d variance between %s and %s is %f \n' % (
                j, proposedMethod, comparedMethod[i], computeVariance(tmp_mcnemarMatrix)))

        mcnemar_mean_mat = calculateMcnemarMeanMatrix(proposedMethod_ScoreFile, proposedMethod_LableFile,
                                                      comparedMethod_ScoreFile[i], comparedMethod_LableFile[i])
        print(mcnemar_mean_mat.reshape([4, ]).tolist())
        print('Mean variance between %s and %s is %f \n' % (
            proposedMethod, comparedMethod[i], computeVariance(mcnemar_mean_mat)))


def main_major_review():
    # Major Review阶段单独计算Concat版本的模型与我们提出的模型间的McNeMar-Test
    proposedMethod = 'fc_attention'
    comparedMethod = ['proposed_concat']

    proposedMethod_ScoreFile = proposedMethod + '_score.txt'
    proposedMethod_LableFile = proposedMethod + '_label.txt'

    comparedMethod_ScoreFile = [x + '_score.txt' for x in comparedMethod]
    comparedMethod_LableFile = [y + '_label.txt' for y in comparedMethod]

    print(comparedMethod_ScoreFile)
    print(comparedMethod_LableFile)

    for i in range(len(comparedMethod_LableFile)):
        for j in range(7):
            tmp_mcnemarMatrix = calculateMcnemarMatrix(proposedMethod_ScoreFile, proposedMethod_LableFile,
                                                       comparedMethod_ScoreFile[i], comparedMethod_LableFile[i], j)
            print(tmp_mcnemarMatrix.reshape([4, ]).tolist())
            # print(tmp_mcnemarMatrix)
            print('Category %d variance between %s and %s is %f \n' % (
                j, proposedMethod, comparedMethod[i], computeVariance(tmp_mcnemarMatrix)))

        mcnemar_mean_mat = calculateMcnemarMeanMatrix(proposedMethod_ScoreFile, proposedMethod_LableFile,
                                                      comparedMethod_ScoreFile[i], comparedMethod_LableFile[i])
        print(mcnemar_mean_mat.reshape([4, ]).tolist())
        print('Mean variance between %s and %s is %f \n' % (
            proposedMethod, comparedMethod[i], computeVariance(mcnemar_mean_mat)))


if __name__ == '__main__':
    # main()
    main_major_review()
