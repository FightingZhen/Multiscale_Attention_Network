import pickle
import numpy as np

data_dir = '../DLD_pickle_data/'
PADDING_SIZE = 4
MIN_VALUE = -1024


def Load_PKL(file_dir):
    with open(file_dir, 'rb') as f:
        message = pickle.load(f)

    return message


def Load_data(data_dir, is_training):
    if is_training:
        tr_img = Load_PKL(data_dir + 'training_image.pkl')
        tr_lab = Load_PKL(data_dir + 'training_label.pkl')

        val_img = Load_PKL(data_dir + 'validation_image.pkl')
        val_lab = Load_PKL(data_dir + 'validation_label.pkl')

        print('-' * 50)
        print('Training sets has %d images, validation sets has %d images' % (tr_img.shape[0], val_img.shape[0]))
        print('-' * 50)

        return tr_img, tr_lab, val_img, val_lab

    else:
        tst_img = Load_PKL(data_dir + 'test_image.pkl')
        tst_lab = Load_PKL(data_dir + 'test_label.pkl')

        print('-' * 50)
        print('Test sets has %d images' % tst_img.shape[0])
        print('-' * 50)

        return tst_img, tst_lab


def random_flip(image_batch):
    for i in range(image_batch.shape[0]):
        flip_prop = np.random.randint(low=0, high=3)
        if flip_prop == 0:
            image_batch[i] = image_batch[i]
        if flip_prop == 1:
            image_batch[i] = np.fliplr(image_batch[i])
        if flip_prop == 2:
            image_batch[i] = np.flipud(image_batch[i])

    return image_batch


def random_crop(image_batch):
    pad_width = ((PADDING_SIZE, PADDING_SIZE), (PADDING_SIZE, PADDING_SIZE), (0, 0))
    new_batch = []
    for i in range(image_batch.shape[0]):
        new_batch.append(image_batch[i])
        new_batch[i] = np.pad(image_batch[i], pad_width=pad_width, mode='constant', constant_values=MIN_VALUE)

        x_offset = np.random.randint(low=0, high=2 * PADDING_SIZE + 1, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * PADDING_SIZE + 1, size=1)[0]
        new_batch[i] = new_batch[i][x_offset:x_offset + 32, y_offset:y_offset + 32, :]

    return new_batch


def shuffle_data(image, label):
    indecies = np.random.permutation(len(image))
    shuffled_image = image[indecies]
    shuffled_label = label[indecies]

    print('Training data shuffled')

    return shuffled_image, shuffled_label


def next_batch(img, label, batch_size, step):
    img_batch = img[step * batch_size:step * batch_size + batch_size]
    lab_batch = label[step * batch_size:step * batch_size + batch_size]

    img_batch = random_flip(img_batch)
    img_batch = random_crop(img_batch)

    return img_batch, lab_batch


if __name__ == '__main__':
    tr_img, tr_lab, val_img, val_lab = Load_data(data_dir, is_training=True)
    tst_img, tst_lab = Load_data(data_dir, is_training=False)

    print(tr_img.mean(), tr_img.std())
    print(val_img.mean(), val_img.std())
    print(tst_img.mean(), tst_img.std())
