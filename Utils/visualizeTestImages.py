import os
import sys

sys.path.append('../Pre_Processing/')
from skimage import io
from skimage import img_as_ubyte
import Init_dld as init
import numpy as np

test_batch_size = 128
CT_WINDOW = 1500
CT_LEVEL = -650

MAX_VALUE = CT_LEVEL + CT_WINDOW / 2
MIN_VALUE = CT_LEVEL - CT_WINDOW / 2

tst_img, tst_lab = init.Load_data(init.data_dir, is_training=False)
tst_batch_num = int(np.ceil(tst_img.shape[0] / test_batch_size))

target_dir = '../visualTestImage/'
counter = 1

if not os.path.isdir(target_dir):
    os.makedirs(target_dir)

for step in range(tst_batch_num):
    tst_x = tst_img[step * test_batch_size:step * test_batch_size + test_batch_size]
    tst_l = tst_lab[step * test_batch_size:step * test_batch_size + test_batch_size]

    for n in range(len(tst_x)):
        tmp_img = tst_x[n].reshape([32, 32])
        sizeX, sizeY = tmp_img.shape

        for j in range(sizeY):
            for i in range(sizeX):
                tmp_img[i][j] = (tmp_img[i][j] - MIN_VALUE) / CT_WINDOW * 255.0

        tmp_img = img_as_ubyte(tmp_img)
        io.imsave('../visualTestImage/' + str(counter) + '.png', tmp_img)
        counter += 1
