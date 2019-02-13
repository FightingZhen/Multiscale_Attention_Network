import pickle
import numpy as np

data_dir = '../DLD_Eigen_pickle_data/'


def Load_PKL(file_dir):
    with open(file_dir, 'rb') as f:
        message = pickle.load(f)

    return message


def Load_data(data_dir, is_training):
    if is_training:
        tr_img = Load_PKL(data_dir + 'training_image.pkl')
        tr_eig = Load_PKL(data_dir + 'training_eigen.pkl')
        tr_lab = Load_PKL(data_dir + 'training_label.pkl')

        val_img = Load_PKL(data_dir + 'validation_image.pkl')
        val_eig = Load_PKL(data_dir + 'validation_eigen.pkl')
        val_lab = Load_PKL(data_dir + 'validation_label.pkl')

        print('-' * 50)
        print('Training sets has %d images, validation sets has %d images' % (tr_img.shape[0], val_img.shape[0]))
        print('-' * 50)

        return tr_img, tr_eig, tr_lab, val_img, val_eig, val_lab

    else:
        tst_img = Load_PKL(data_dir + 'test_image.pkl')
        tst_eig = Load_PKL(data_dir + 'test_eigen.pkl')
        tst_lab = Load_PKL(data_dir + 'test_label.pkl')

        print('-' * 50)
        print('Test sets has %d images' % tst_img.shape[0])
        print('-' * 50)

        return tst_img, tst_eig, tst_lab


def random_flip(image_batch, eigen_batch):
    for i in range(image_batch.shape[0]):
        flip_prop = np.random.randint(low=0, high=3)
        if flip_prop == 0:
            image_batch[i] = image_batch[i]
            eigen_batch[i] = eigen_batch[i]
        if flip_prop == 1:
            image_batch[i] = np.fliplr(image_batch[i])
            eigen_batch[i] = np.fliplr(eigen_batch[i])
        if flip_prop == 2:
            image_batch[i] = np.flipud(image_batch[i])
            eigen_batch[i] = np.flipud(eigen_batch[i])

    return image_batch, eigen_batch


def shuffle_data(image, eigen, label):
    indecies = np.random.permutation(len(image))
    shuffled_image = image[indecies]
    shuffled_eigen = eigen[indecies]
    shuffled_label = label[indecies]

    print('Training data shuffled')

    return shuffled_image, shuffled_eigen, shuffled_label


def next_batch(img, eigen, label, batch_size, step):
    img_batch = img[step * batch_size:step * batch_size + batch_size]
    eig_batch = eigen[step * batch_size:step * batch_size + batch_size]
    lab_batch = label[step * batch_size:step * batch_size + batch_size]

    img_batch, eig_batch = random_flip(img_batch, eig_batch)

    return img_batch, eig_batch, lab_batch


if __name__ == '__main__':
    x, xe, y, vx, ve, vy = Load_data(data_dir, is_training=True)
