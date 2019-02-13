import numpy as np
import FileNameExtraction_DLD as FNE
import pickle
import os

MIN_VALUE = -1282
PADDING_SIZE = 2

save_dir = 'D:/DLD_pickle_data/'
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


def collectInformation(fileNameClass, file_dir, mode):
    if mode == 'Train':
        training_image, training_label = [], []
        for single_class in fileNameClass:
            class_number = 0
            class_name = ''
            for single_fileName in single_class:
                temp_image, temp_label = extractInformation(single_fileName, file_dir)
                class_number += len(temp_image)
                class_name = str(temp_label[0])
                for item_temp_image, item_temp_label in zip(temp_image, temp_label):
                    training_image.append(item_temp_image)
                    training_label.append(item_temp_label)
            print('Training Sets class [%s] have [%g] images' % (class_name, class_number))

        num_img = len(training_image)
        training_lab_onehot = np.zeros((num_img, 7))
        for i in range(len(training_label)):
            training_lab_onehot[i][training_label[i] - 1] += 1
        training_img = np.reshape(training_image, (num_img, 32, 32, 1)).astype(np.float32)

        return training_img, training_lab_onehot

    if mode == 'Validation':
        validation_image, validation_label = [], []
        for single_class in fileNameClass:
            class_number = 0
            class_name = ''
            for single_fileName in single_class:
                temp_image, temp_label = extractInformation(single_fileName, file_dir)
                class_number += len(temp_image)
                class_name = str(temp_label[0])
                for item_temp_image, item_temp_label in zip(temp_image, temp_label):
                    validation_image.append(item_temp_image)
                    validation_label.append(item_temp_label)
            print('Validation Sets class [%s] have [%g] images' % (class_name, class_number))

        num_img = len(validation_image)
        validation_lab_onehot = np.zeros((num_img, 7))
        for i in range(len(validation_label)):
            validation_lab_onehot[i][validation_label[i] - 1] += 1
        validation_img = np.reshape(validation_image, (num_img, 32, 32, 1)).astype(np.float32)

        return validation_img, validation_lab_onehot

    if mode == 'Test':
        test_image, test_label = [], []
        for single_class in fileNameClass:
            class_number = 0
            class_name = ''
            for single_fileName in single_class:
                temp_image, temp_label = extractInformation(single_fileName, file_dir)
                class_number += len(temp_image)
                class_name = str(temp_label[0])
                for item_temp_image, item_temp_label in zip(temp_image, temp_label):
                    test_image.append(item_temp_image)
                    test_label.append(item_temp_label)
            print('Test Sets class [%s] have [%g] images' % (class_name, class_number))

        num_img = len(test_image)
        test_lab_onehot = np.zeros((num_img, 7))
        for i in range(len(test_label)):
            test_lab_onehot[i][test_label[i] - 1] += 1
        test_img = np.reshape(test_image, (num_img, 32, 32, 1)).astype(np.float32)

        return test_img, test_lab_onehot


def Initialization():
    trainingFileSets = fileNameClasses_training
    validationFileSets = fileNameClasses_validation

    training_x, training_y = collectInformation(trainingFileSets, FNE.training_path, mode='Train')
    validation_x, validation_y = collectInformation(validationFileSets, FNE.validation_path, mode='Validation')

    print('-' * 25 + 'Processing training & validation data finished !' + '-' * 25)

    training_size = training_x.shape[0]
    validation_size = validation_x.shape[0]

    print('-' * 10 + 'Number of training data : %d, validation data : %d' % (
        training_size, validation_size) + '-' * 10)

    # print('training set mean : ' + str(np.mean(training_x)))
    # print('training set stddev : ' + str(np.std(training_x)))
    #
    # print('validation set mean : ' + str(np.mean(validation_x)))
    # print('validation set stddev : ' + str(np.std(validation_x)))

    pick2disk(training_x, save_dir + 'training_image.pkl')
    pick2disk(training_y, save_dir + 'training_label.pkl')
    pick2disk(validation_x, save_dir + 'validation_image.pkl')
    pick2disk(validation_y, save_dir + 'validation_label.pkl')

    testFileSets = fileNameClasses_test

    test_x, test_y = collectInformation(testFileSets, FNE.test_path, mode='Test')

    print('-' * 25 + 'Processing test data finished !' + '-' * 25)

    test_size = test_x.shape[0]

    print('-' * 10 + 'Number of test data %d' % test_size + '-' * 10)

    # print('test set mean : ' + str(np.mean(test_x)))
    # print('test set stddev : ' + str(np.std(test_x)))

    pick2disk(test_x, save_dir + 'test_image.pkl')
    pick2disk(test_y, save_dir + 'test_label.pkl')


def pick2disk(input_array, file_dir):
    with open(file_dir, 'wb') as f:
        pickle.dump(input_array, f)


def loadFromPKL(file_dir):
    with open(file_dir, 'rb') as f:
        message = pickle.load(f)

    return message


if __name__ == '__main__':
    Initialization()
    # mess = loadFromPKL('D:/DLD_pickle_data/training_label.pkl')