import os
import operator

image_dir = 'DLD_o8_s4'
eigen_dir = 'DLD_Eigens'

training_path = "D:/" + image_dir + "/train/"
validation_path = "D:/" + image_dir + "/validation/"
test_path = "D:/" + image_dir + "/test/"

eigen_training_path = "D:/" + eigen_dir + "/train/"
eigen_validation_path = "D:/" + eigen_dir + "/validation/"
eigen_test_path = "D:/" + eigen_dir + "/test/"


def extractFiles(path, mode):
    if mode == 'Train':
        print('Training Data dir:', training_path)
    if mode == 'Validation':
        print('Validation Data dir:', validation_path)
    if mode == 'Test':
        print('Test Data dir:', test_path)

    file_Names = os.listdir(path)
    temp_File_Names = []
    for f in file_Names:
        temp_File_Names.append(f[:-8])
    opacity_class = sorted(list(set(temp_File_Names)))

    filesInClasses = []
    for c in opacity_class:
        # print(c)
        ind = getStrIndex(temp_File_Names, c)
        # print(ind)
        filesInClasses.append([file_Names[i] for i in ind])
        # print(filesInClasses)
        # print(filesInClasses)
    # for f in filesInClasses:
    #     print(f)
    return filesInClasses


def getStrIndex(strList, str):
    ind = []
    for i in range(len(strList)):
        if operator.eq(strList[i], str) == 1:
            ind.append(i)
    return ind


if __name__ == '__main__':
    extractFiles(training_path, mode='Train')
