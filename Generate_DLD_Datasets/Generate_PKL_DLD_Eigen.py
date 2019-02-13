import numpy as np
import FileNameExtraction_DLD_Eigen as FNE
import pickle
import os

MIN_VALUE = -1282
PADDING_SIZE = 2

save_dir = 'D:/DLD_Eigen_pickle_data/'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

fileNameClasses_training = FNE.extractFiles(FNE.training_path, mode='Train')
fileNameClasses_validation = FNE.extractFiles(FNE.validation_path, mode='Validation')
fileNameClasses_test = FNE.extractFiles(FNE.test_path, mode='Test')


def extractInformation(fileName, file_dir):
    image, label = [], []
    temp_image = []
    line_counter = 0

    file = open(file_dir + fileName)
    for line in file:
        if line.index('\n') == 1:
            label.append(int(line))
            continue
        else:
            str_line = line.strip(',\n')
            split_str = str_line.split(',')
            temp_image = temp_image + list(map(float, split_str))
            line_counter += 1
            if line_counter % 32 == 0:
                image.append(temp_image)
                temp_image = []
    file.close()

    return image, label


def collectInformation(fileNameClass, file_dir, eigen_file_dir, mode):
    if mode == 'Train':
        training_image, eigen_training_image, training_label = [], [], []
        for single_class in fileNameClass:
            class_number = 0
            class_name = ''
            for single_fileName in single_class:
                temp_image, temp_label = extractInformation(single_fileName, file_dir)
                temp_eigen_image, _ = extractInformation(single_fileName, eigen_file_dir)
                class_number += len(temp_image)
                class_name = str(temp_label[0])
                for item_temp_image, item_temp_eigen_image, item_temp_label in zip(temp_image, temp_eigen_image,
                                                                                   temp_label):
                    training_image.append(item_temp_image)
                    eigen_training_image.append(item_temp_eigen_image)
                    training_label.append(item_temp_label)
            print('Training Sets class %s have %g images' % (class_name, class_number))

        num_img = len(training_image)
        training_lab_onehot = np.zeros((num_img, 7))
        for i in range(len(training_label)):
            training_lab_onehot[i][training_label[i] - 1] += 1
        training_img = np.reshape(training_image, (num_img, 32, 32, 1)).astype(np.float32)
        training_eig = np.reshape(eigen_training_image, (num_img, 32, 32, 3)).astype(np.float32)

        return training_img, training_eig, training_lab_onehot

    if mode == 'Validation':
        validation_image, eigen_validation_image, validation_label = [], [], []
        for single_class in fileNameClass:
            class_number = 0
            class_name = ''
            for single_fileName in single_class:
                temp_image, temp_label = extractInformation(single_fileName, file_dir)
                temp_eigen_image, _ = extractInformation(single_fileName, eigen_file_dir)
                class_number += len(temp_image)
                class_name = str(temp_label[0])
                for item_temp_image, item_temp_eigen_image, item_temp_label in zip(temp_image, temp_eigen_image,
                                                                                   temp_label):
                    validation_image.append(item_temp_image)
                    eigen_validation_image.append(item_temp_eigen_image)
                    validation_label.append(item_temp_label)
            print('Validation Sets class %s have %g images' % (class_name, class_number))

        num_img = len(validation_image)
        validation_lab_onehot = np.zeros((num_img, 7))
        for i in range(len(validation_label)):
            validation_lab_onehot[i][validation_label[i] - 1] += 1
        validation_img = np.reshape(validation_image, (num_img, 32, 32, 1)).astype(np.float32)
        validation_eig = np.reshape(eigen_validation_image, (num_img, 32, 32, 3)).astype(np.float32)

        return validation_img, validation_eig, validation_lab_onehot

    if mode == 'Test':
        test_image, eigen_test_image, test_label = [], [], []
        for single_class in fileNameClass:
            class_number = 0
            class_name = ''
            for single_fileName in single_class:
                temp_image, temp_label = extractInformation(single_fileName, file_dir)
                temp_eigen_image, _ = extractInformation(single_fileName, eigen_file_dir)
                class_number += len(temp_image)
                class_name = str(temp_label[0])
                for item_temp_image, item_temp_eigen_image, item_temp_label in zip(temp_image, temp_eigen_image,
                                                                                   temp_label):
                    test_image.append(item_temp_image)
                    eigen_test_image.append(item_temp_eigen_image)
                    test_label.append(item_temp_label)
            print('Test Sets class %s have %g images' % (class_name, class_number))

        num_img = len(test_image)
        test_lab_onehot = np.zeros((num_img, 7))
        for i in range(len(test_label)):
            test_lab_onehot[i][test_label[i] - 1] += 1
        test_img = np.reshape(test_image, (num_img, 32, 32, 1)).astype(np.float32)
        test_eig = np.reshape(eigen_test_image, (num_img, 32, 32, 3)).astype(np.int32)

        return test_img, test_eig, test_lab_onehot


def Initialization():
    trainingFileSets = fileNameClasses_training
    validationFileSets = fileNameClasses_validation
    testFileSets = fileNameClasses_test

    training_img, training_eig, training_lab = collectInformation(trainingFileSets, FNE.training_path,
                                                                  FNE.eigen_training_path, mode='Train')
    validation_img, validation_eig, validation_lab = collectInformation(validationFileSets, FNE.validation_path,
                                                                        FNE.eigen_validation_path,
                                                                        mode='Validation')
    test_img, test_eig, test_lab = collectInformation(testFileSets, FNE.test_path, FNE.eigen_test_path, mode='Test')

    print('-' * 25 + 'Processing three kinds of data finished !' + '-' * 25)

    training_size = training_img.shape[0]
    validation_size = validation_img.shape[0]
    test_size = test_img.shape[0]

    print('-' * 10 + 'Number of training data : %d, validation data : %d, test data %d' % (
        training_size, validation_size, test_size) + '-' * 10)

    pick2disk(training_img, save_dir + 'training_image.pkl')
    pick2disk(training_eig, save_dir + 'training_eigen.pkl')
    pick2disk(training_lab, save_dir + 'training_label.pkl')

    pick2disk(validation_img, save_dir + 'validation_image.pkl')
    pick2disk(validation_eig, save_dir + 'validation_eigen.pkl')
    pick2disk(validation_lab, save_dir + 'validation_label.pkl')

    pick2disk(test_img, save_dir + 'test_image.pkl')
    pick2disk(test_eig, save_dir + 'test_eigen.pkl')
    pick2disk(test_lab, save_dir + 'test_label.pkl')


def random_flip(image, eigen_value):
    flip_image = []
    flip_eigen = []

    image = np.reshape(image, [32, 32])
    eigen = np.reshape(eigen_value, [32, 32, 3])

    flip_prop = np.random.randint(low=0, high=3)
    if flip_prop == 0:
        flip_image = image
        flip_eigen = eigen
    if flip_prop == 1:
        flip_image = np.fliplr(image)
        flip_eigen = np.fliplr(eigen)
    if flip_prop == 2:
        flip_image = np.flipud(image)
        flip_eigen = np.flipud(eigen)

    flip_image = np.reshape(flip_image, [32 * 32]).tolist()
    flip_eigen = np.reshape(flip_eigen, [32 * 32 * 3]).tolist()

    return flip_image, flip_eigen


def pick2disk(input_array, file_dir):
    with open(file_dir, 'wb') as f:
        pickle.dump(input_array, f)


def loadFromPKL(file_dir):
    with open(file_dir, 'rb') as f:
        message = pickle.load(f)

    return message


if __name__ == '__main__':
    Initialization()
    # mess = loadFromPKL('D:/DLD_Eigen_pickle_data/training_eigen.pkl')
    # print(mess.shape)