import sys

sys.path.append('../Pre_Processing/')
import matplotlib.pyplot as plt
import Init_dld as init
import numpy as np
import time
import os
import argparse
import tensorflow as tf
import tensorflow.contrib.layers as layers
from skimage import io
from skimage import img_as_ubyte
import cv2

plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument("-model")
parser.add_argument("-gpu")
parser.add_argument("-epoch", type=int, default=200)
parser.add_argument("-ksize", type=int, default=3)
parser.add_argument("-lr", type=float, default=0.01)
parser.add_argument("-weight_decay", type=float, default=1e-4)
parser.add_argument("-momentum", type=float, default=0.9)
parser.add_argument("-decay_rate", type=float, default=0.97)
parser.add_argument("-best_epoch", default='0')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# --------------------hyper parameters-------------------- #
Training = False
augmentation = True

NUM_LABELS = 7
IMAGE_SIZE = 32

num_block1 = 1
num_block2 = 2
num_block3 = 1

training_batch_size = 128
validation_batch_size = 128
test_batch_size = 128

epoch = args.epoch
momentum = args.momentum
decay_rate = args.decay_rate
weight_decay = args.weight_decay
lr = args.lr
ksize = args.ksize
# -------------------------------------------------------- #

pulmonary_category = {0: 'CON',
                      1: 'MUL_GGO',
                      2: 'HCM',
                      3: 'RET_GGO',
                      4: 'EMP',
                      5: 'NOD',
                      6: 'NOR'}

Load_Data = not Training
model_name = args.model

checkpoint_dir = '../Checkpoint/' + model_name + '/'
checkpoint_test_dir = '../Checkpoint/' + model_name + '/test/'
best_epoch = args.best_epoch

tr_img, tr_lab, val_img, val_lab = init.Load_data(init.data_dir, is_training=True)
tst_img, tst_lab = init.Load_data(init.data_dir, is_training=False)

tr_iteration = tr_img.shape[0] // training_batch_size

cfg_information = ''
cfg_information += '-' * 50 + '\n'
cfg_information += 'Epoch:[%g]' % epoch + '\n'
cfg_information += 'Model_name:[%s]' % model_name + '\n'
cfg_information += 'Learning_rate:[%g]' % lr + '\n'
cfg_information += 'Kernel_size:[%g]' % ksize + '\n'
cfg_information += 'Image_size:[%g]' % IMAGE_SIZE + '\n'
cfg_information += 'Augmentation:[%s]' % str(augmentation) + '\n'
cfg_information += 'Num_block1:[%g]' % num_block1 + '\n'
cfg_information += 'Num_block2:[%g]' % num_block2 + '\n'
cfg_information += 'Num_block3:[%g]' % num_block3 + '\n'
cfg_information += 'Total_layers:[%s]' % str((num_block1 * 2 + num_block2 * 2 + num_block3 * 2) + 2) + '\n'
cfg_information += '-' * 50 + '\n'

epoch_plt = []
training_accuracy_plt = []
validation_accuracy_plt = []
training_loss_plt = []
validation_loss_plt = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def plot_acc_loss():
    plt.figure(figsize=(10.24, 7.68))
    plt.plot(epoch_plt, training_accuracy_plt, linewidth=1.0, linestyle='-', label='training_accuracy')
    plt.plot(epoch_plt, validation_accuracy_plt, linewidth=1.0, color='red', linestyle='--',
             label='validation_accuracy')
    plt.title('Accuracy')
    plt.ylim([0.0, 1.2])
    plt.xticks([x for x in range(0, epoch + 1, 10)])
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(checkpoint_dir + 'Accuracy.png')

    plt.figure(figsize=(10.24, 7.68))
    plt.plot(epoch_plt, training_loss_plt, linewidth=1.0, linestyle='-', label='training_loss')
    plt.plot(epoch_plt, validation_loss_plt, color='red', linewidth=1.0, linestyle='--', label='validation_loss')
    plt.title('Loss')
    plt.ylim([0.0, 3.0])
    plt.xticks([x for x in range(0, epoch + 1, 10)])
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(checkpoint_dir + 'Loss.png')


def save2file(log_info):
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    logfile = open(checkpoint_dir + model_name + '.txt', 'a+')
    print(log_info)
    print(log_info, file=logfile)
    logfile.close()


def conv_layer(input, out_channel, ksize, stride, layer_name):
    with tf.variable_scope(layer_name):
        weight = tf.get_variable('weight', [ksize, ksize, input.get_shape()[-1], out_channel],
                                 initializer=layers.variance_scaling_initializer(),
                                 regularizer=layers.l2_regularizer(weight_decay))
        result = tf.nn.conv2d(input, weight, strides=[1, stride, stride, 1], padding='SAME')

        tf.summary.histogram('weight', weight)
        tf.summary.histogram('result', result)

    return result


def relu_layer(input, layer_name):
    with tf.variable_scope(layer_name):
        return tf.nn.relu(input)


def bn_layer(input, layer_name):
    with tf.variable_scope(layer_name):
        return tf.layers.batch_normalization(input, training=is_training)


def avg_pool_layer(input, ksize, stride, layer_name):
    with tf.variable_scope(layer_name):
        return tf.nn.avg_pool(input,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')


def global_avg_pool_layer(input, layer_name):
    with tf.variable_scope(layer_name):
        width = input.get_shape().as_list()[1]
        return tf.nn.avg_pool(input,
                              ksize=[1, width, width, 1],
                              strides=[1, width, width, 1],
                              padding='VALID')


def flatten(input, layer_name):
    with tf.variable_scope(layer_name):
        return tf.layers.flatten(input)


def attention_block(input, block_name):
    with tf.variable_scope(block_name):
        in_channel = input.get_shape().as_list()[-1]
        bn1 = bn_layer(input, layer_name='bn1')
        relu1 = relu_layer(bn1, layer_name='relu1')
        global_pool = global_avg_pool_layer(input=relu1, layer_name='global_pool')
        global_pool = flatten(global_pool, layer_name='flatten')
        fc1 = fc_layer(input=global_pool, out_channel=in_channel // 2, layer_name='fc1')
        relu2 = relu_layer(fc1, layer_name='relu2')
        fc2 = fc_layer(relu2, out_channel=in_channel, layer_name='fc2')
        sigmoid = tf.nn.sigmoid(fc2)
        tf.summary.histogram('attention_sigmoid', sigmoid)

        sig_reshape = tf.reshape(sigmoid, [-1, 1, 1, in_channel])
        attention_mask = input * sig_reshape
        image_summary(attention_mask, layer_name='attention_mask')
        attention_residual = input + attention_mask

    return attention_residual, sigmoid


def image_summary(input, layer_name, input_img=False):
    with tf.variable_scope(layer_name):
        out_channel = input.get_shape().as_list()[-1]
        if input_img:
            tf.summary.image(layer_name, tf.expand_dims(input[:, :, :, out_channel // 2], axis=3), max_outputs=1)
        else:
            tf.summary.image(layer_name, tf.expand_dims(input[:, :, :, out_channel // 2], axis=3), max_outputs=1)
            tf.summary.image(layer_name, tf.expand_dims(input[:, :, :, out_channel // 2 - 5], axis=3), max_outputs=1)
            tf.summary.image(layer_name, tf.expand_dims(input[:, :, :, out_channel // 2 + 5], axis=3), max_outputs=1)


def fc_layer(input, out_channel, layer_name):
    with tf.variable_scope(layer_name):
        in_channel = input.get_shape().as_list()[-1]
        weight = tf.get_variable('weight', [in_channel, out_channel],
                                 initializer=layers.variance_scaling_initializer(),
                                 regularizer=layers.l2_regularizer(weight_decay))

        bias = tf.get_variable('bias', [out_channel], initializer=tf.zeros_initializer())
        fc = tf.matmul(input, weight) + bias

        tf.summary.histogram('weight', weight)
        tf.summary.histogram('bias', bias)
        tf.summary.histogram('fc_result', fc)

    return fc


def partial_fc_layer(input, out_channel, layer_name):
    with tf.variable_scope(layer_name):
        partial_fc_bn = bn_layer(input, layer_name='partial_fc_bn')
        partial_fc_relu = relu_layer(partial_fc_bn, layer_name='partial_fc_relu')
        partial_fc_pool = global_avg_pool_layer(partial_fc_relu, layer_name='global_pool')
        partial_fc_pool = flatten(partial_fc_pool, layer_name='flatten')
        partial_fc = fc_layer(partial_fc_pool, out_channel, layer_name='partial_fc')

    return partial_fc


def residual_unit(input, out_channel, ksize, unit_name, down_sampling, first_conv=False):
    in_channel = input.get_shape().as_list()[-1]
    if down_sampling:
        stride = 2
        increase_dim = True
    else:
        stride = 1
        increase_dim = False

    with tf.variable_scope(unit_name):
        if first_conv:
            conv1 = conv_layer(input=input,
                               out_channel=out_channel,
                               ksize=ksize,
                               stride=stride,
                               layer_name='conv1')
            image_summary(conv1, layer_name='image_summary_conv1')
        else:
            conv1 = bn_layer(input=input, layer_name='conv1_bn')
            conv1 = relu_layer(input=conv1, layer_name='conv1_activation')
            conv1 = conv_layer(input=conv1,
                               out_channel=out_channel,
                               ksize=ksize,
                               stride=stride,
                               layer_name='conv1')
            image_summary(conv1, layer_name='image_summary_conv1')

        conv2 = bn_layer(input=conv1, layer_name='conv2_bn')
        conv2 = relu_layer(input=conv2, layer_name='conv2_activation')
        conv2 = conv_layer(input=conv2,
                           out_channel=out_channel,
                           ksize=ksize,
                           stride=1,
                           layer_name='conv2')
        image_summary(conv2, layer_name='image_summary_conv2')

        if increase_dim is True:
            identical_map = avg_pool_layer(input, ksize=2, stride=2, layer_name='identical_pool')
            identical_map = tf.pad(identical_map, [[0, 0], [0, 0], [0, 0],
                                                   [(out_channel - in_channel) // 2, (out_channel - in_channel) // 2]])
        else:
            identical_map = input

        added = tf.add(conv2, identical_map)

    return added


def section(input, num, out_channel, ksize, section_name, down_sampling, first_conv):
    out = input
    with tf.variable_scope(section_name):
        out = residual_unit(input=out,
                            out_channel=out_channel,
                            ksize=ksize,
                            unit_name='unit_1',
                            first_conv=first_conv,
                            down_sampling=down_sampling)
        for i in range(2, num + 1):
            out = residual_unit(input=out,
                                out_channel=out_channel,
                                ksize=ksize,
                                unit_name='unit' + str(i),
                                first_conv=False,
                                down_sampling=False)

        final_out = out

        attention_out, sig_weight = attention_block(final_out, block_name='attention_block')

    return attention_out, final_out, sig_weight


def inference(input,
              ksize,
              out_channel1,
              out_channel2,
              out_channel3):
    with tf.variable_scope('unit0'):
        unit0 = conv_layer(input=input,
                           out_channel=out_channel1,
                           ksize=ksize,
                           stride=1,
                           layer_name='conv')
        image_summary(unit0, layer_name='unit0_image')
        unit0 = bn_layer(unit0, layer_name='bn')
        unit0 = relu_layer(unit0, layer_name='activation')

    attention_unit0, sig_weight0 = attention_block(unit0, block_name='attention_unit0')

    section1_attention, sec1_attention_input, sec1_sig_weight = section(input=attention_unit0,
                                                                        out_channel=out_channel1,
                                                                        num=num_block1,
                                                                        ksize=ksize,
                                                                        section_name='section1',
                                                                        first_conv=True,
                                                                        down_sampling=False)

    partial_fc_sec1 = partial_fc_layer(section1_attention, out_channel=NUM_LABELS, layer_name='partial_fc_section1')

    section2_attention, sec2_attention_input, sec2_sig_weight = section(input=section1_attention,
                                                                        out_channel=out_channel2,
                                                                        num=num_block2,
                                                                        ksize=ksize,
                                                                        section_name='section2',
                                                                        first_conv=False,
                                                                        down_sampling=True)

    partial_fc_sec2 = partial_fc_layer(section2_attention, out_channel=NUM_LABELS, layer_name='partial_fc_section2')

    section3_attention, sec3_attention_input, sec3_sig_weight = section(input=section2_attention,
                                                                        out_channel=out_channel3,
                                                                        num=num_block3,
                                                                        ksize=ksize,
                                                                        section_name='section3',
                                                                        first_conv=False,
                                                                        down_sampling=True)

    partial_fc_sec3 = partial_fc_layer(section3_attention, out_channel=NUM_LABELS, layer_name='partial_fc_section3')

    added = tf.add_n([partial_fc_sec1, partial_fc_sec2, partial_fc_sec3])

    return added, sec1_attention_input, sec1_sig_weight, [[sec1_attention_input, partial_fc_sec1],
                                                          [sec2_attention_input, partial_fc_sec2],
                                                          [sec3_attention_input, partial_fc_sec3]]


def f_value(matrix):
    f = 0.0
    length = len(matrix[0])
    for i in range(length):
        recall = matrix[i][i] / np.sum([matrix[i][m] for m in range(NUM_LABELS)])
        precision = matrix[i][i] / np.sum([matrix[n][i] for n in range(NUM_LABELS)])
        result = (recall * precision) / (recall + precision)
        f += result
    f *= (2 / NUM_LABELS)
    return f


def validation_procedure(loss, val_img, val_lab):
    confusion_matrics = np.zeros([NUM_LABELS, NUM_LABELS], dtype="int")
    val_loss = 0.0

    val_batch_num = int(np.ceil(val_img.shape[0] / validation_batch_size))
    for step in range(val_batch_num):
        val_x = val_img[step * validation_batch_size:step * validation_batch_size + validation_batch_size]
        val_l = val_lab[step * validation_batch_size:step * validation_batch_size + validation_batch_size]

        [matrix_row, matrix_col], tmp_loss = sess.run([distribution, loss],
                                                      feed_dict={x: val_x,
                                                                 y_: val_l,
                                                                 is_training: False})
        for m, n in zip(matrix_row, matrix_col):
            confusion_matrics[m][n] += 1

        val_loss += tmp_loss

    validation_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(NUM_LABELS)])) / float(
        np.sum(confusion_matrics))
    validation_loss = val_loss / val_batch_num

    return validation_accuracy, validation_loss


def test_procedure(tst_img, tst_lab):
    confusion_matrics = np.zeros([NUM_LABELS, NUM_LABELS], dtype="int")

    tst_batch_num = int(np.ceil(tst_img.shape[0] / test_batch_size))
    for step in range(tst_batch_num):
        # print('Tst_batch_num' + str(tst_batch_num))
        tst_x = tst_img[step * test_batch_size:step * test_batch_size + test_batch_size]
        tst_l = tst_lab[step * test_batch_size:step * test_batch_size + test_batch_size]

        matrix_row, matrix_col = sess.run(distribution,
                                          feed_dict={x: tst_x,
                                                     y_: tst_l,
                                                     is_training: False})
        for m, n in zip(matrix_row, matrix_col):
            confusion_matrics[m][n] += 1

    test_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(NUM_LABELS)])) / float(
        np.sum(confusion_matrics))
    detail_test_accuracy = [confusion_matrics[i][i] / np.sum(confusion_matrics[i]) for i in range(NUM_LABELS)]
    log1 = "Test Accuracy : %g" % test_accuracy
    log2 = np.array(confusion_matrics.tolist())
    log3 = ''
    for j in range(NUM_LABELS):
        log3 += 'category %s test accuracy : %g\n' % (pulmonary_category[j], detail_test_accuracy[j])
    log4 = 'F_Value : %g' % f_value(confusion_matrics)

    save2file(log1)
    save2file(log2)
    save2file(log3)
    save2file(log4)


def plotROCInformation(tst_img, tst_lab):
    tst_batch_num = int(np.ceil(tst_img.shape[0] / test_batch_size))

    for step in range(tst_batch_num):
        print('Tst_batch_num' + str(tst_batch_num))
        tst_x = tst_img[step * test_batch_size:step * test_batch_size + test_batch_size]
        tst_l = tst_lab[step * test_batch_size:step * test_batch_size + test_batch_size]

        y_score = sess.run(y_conv, feed_dict={x: tst_x,
                                              y_: tst_l,
                                              is_training: False})

        with open('../fc_attention_test_label.txt', 'a+') as f:
            for i in range(tst_l.shape[0]):
                tmp_str = ''
                for item in tst_l[i]:
                    tmp_str += (str(item) + ' ')
                f.write(tmp_str + '\n')

        with open('../fc_attention_y_score.txt', 'a+') as g:
            for i in range(y_score.shape[0]):
                tmp_str = ''
                for item in y_score[i]:
                    tmp_str += (str(item) + ' ')
                g.write(tmp_str + '\n')


def saveImages(imageSet, out_dir, step_num, sig_weight, origin):
    setShape = imageSet.shape
    batch_number = 1
    channel_number = 1

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for bat in range(0, test_batch_size, 8):
        for cha in range(0, setShape[3], 8):
            tmp_img = imageSet[bat, :, :, cha].reshape([32, 32])
            tmp_weight = sig_weight[bat, cha]
            sizeX, sizeY = tmp_img.shape

            if origin:
                MAX_VALUE = 100
                MIN_VALUE = -1400
            else:
                MAX_VALUE = np.max(tmp_img)
                MIN_VALUE = np.min(tmp_img)

            for j in range(sizeY):
                for i in range(sizeX):
                    tmp_img[i][j] = (tmp_img[i][j] - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * 255.0

            tmp_img = tmp_weight * tmp_img
            tmp_img = tmp_img.astype(np.uint8)

            io.imsave(out_dir + str(step_num) + '_' + str(batch_number) + '_' + str(channel_number) + '_' + str(
                tmp_weight) + '.png', tmp_img)

            print('File ' + out_dir + str(step_num) + '_' + str(batch_number) + '_' + str(
                channel_number) + '.png' + ' finished')
            channel_number += 1
        batch_number += 1
        channel_number = 1


def attentionVisualization(tst_img):
    tst_batch_num = int(np.ceil(tst_img.shape[0] / test_batch_size))

    for step in range(tst_batch_num):
        print('step : ' + str(step))
        tst_x = tst_img[step * test_batch_size:step * test_batch_size + test_batch_size]

        # save origin test image
        original_image_save_dir = '../attentionImage_new/origin_image/'
        saveImages(tst_x, original_image_save_dir, step, sig_weight=np.ones([128, 128], dtype=np.int32), origin=True)

        # save inputs and outputs of attention block
        attentionInput, sig_weight = sess.run([sec1_att_input, sec1_sig_weight],
                                              feed_dict={x: tst_x, is_training: False})
        attentionInputDir = '../attentionImage_new/attention_input/'
        attentionOutputDir = '../attentionImage_new/attention_output/'

        saveImages(attentionInput, attentionInputDir, step, sig_weight=np.ones([128, 128], dtype=np.int32), origin=False)
        saveImages(attentionInput, attentionOutputDir, step, sig_weight=sig_weight, origin=False)


def pre_grad_cam():
    predicted_class = tf.arg_max(tf.nn.softmax(y_conv), 1)

    conv_layer_cam1 = netLayerSets[0][0]
    conv_layer_cam2 = netLayerSets[1][0]
    conv_layer_cam3 = netLayerSets[2][0]

    y_score_cam1 = netLayerSets[0][1]
    y_score_cam2 = netLayerSets[1][1]
    y_score_cam3 = netLayerSets[2][1]

    one_hot = tf.one_hot(indices=predicted_class, depth=NUM_LABELS)
    signal_cam1 = tf.multiply(y_score_cam1, one_hot)
    signal_cam2 = tf.multiply(y_score_cam2, one_hot)
    signal_cam3 = tf.multiply(y_score_cam3, one_hot)

    loss_cam1 = tf.reduce_mean(signal_cam1)
    loss_cam2 = tf.reduce_mean(signal_cam2)
    loss_cam3 = tf.reduce_mean(signal_cam3)

    grads_cam1 = tf.gradients(loss_cam1, conv_layer_cam1)[0]
    grads_cam2 = tf.gradients(loss_cam2, conv_layer_cam2)[0]
    grads_cam3 = tf.gradients(loss_cam3, conv_layer_cam3)[0]

    norm_grads_cam1 = tf.div(grads_cam1, tf.sqrt(tf.reduce_mean(tf.square(grads_cam1))) + tf.constant(1e-5))
    norm_grads_cam2 = tf.div(grads_cam2, tf.sqrt(tf.reduce_mean(tf.square(grads_cam2))) + tf.constant(1e-5))
    norm_grads_cam3 = tf.div(grads_cam3, tf.sqrt(tf.reduce_mean(tf.square(grads_cam3))) + tf.constant(1e-5))

    return [conv_layer_cam1, conv_layer_cam2, conv_layer_cam3], [norm_grads_cam1, norm_grads_cam2, norm_grads_cam3]


def grad_cam(input, conv_cam, norm_gradient):
    output, grads_val = sess.run([conv_cam, norm_gradient], feed_dict={x: input, is_training: False})
    output = output[0]
    grads_val = grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (32, 32), interpolation=cv2.INTER_CUBIC)
    cam = cam / np.max(cam)

    return cam


def saveCAM_Image(inputMap, outdir, fileCounter):
    fig, ax = plt.subplots()
    ax.imshow(inputMap, cmap=plt.cm.jet, alpha=1.0, interpolation='nearest', vmin=0, vmax=1)
    plt.axis('off')

    height, width = inputMap.shape

    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(outdir + str(fileCounter) + '.png', dpi=300)
    plt.close()


def generate_GradCAM_Image(save_dir='../Grad_CAM_Split/'):
    save_dir_sc1 = save_dir + 'scale1/'
    save_dir_sc2 = save_dir + 'scale2/'
    save_dir_sc3 = save_dir + 'scale3/'
    save_dir_sum = save_dir + 'sum/'

    if not os.path.isdir(save_dir_sc1):
        os.makedirs(save_dir_sc1)
    if not os.path.isdir(save_dir_sc2):
        os.makedirs(save_dir_sc2)
    if not os.path.isdir(save_dir_sc3):
        os.makedirs(save_dir_sc3)
    if not os.path.isdir(save_dir_sum):
        os.makedirs(save_dir_sum)

    tst_batch_num = int(np.ceil(tst_img.shape[0] / test_batch_size))

    out_counter = 1

    for step in range(tst_batch_num):
        tst_x = tst_img[step * test_batch_size:step * test_batch_size + test_batch_size]

        for m in range(0, tst_x.shape[0]):
            imgForCal = np.expand_dims(tst_x[m], 0)

            cam3_1 = grad_cam(imgForCal, conv_cam[0], y_cam[0])
            cam3_2 = grad_cam(imgForCal, conv_cam[1], y_cam[1])
            cam3_3 = grad_cam(imgForCal, conv_cam[2], y_cam[2])
            cam3_sum = cam3_1 + cam3_2 + cam3_3

            cam3_1 /= cam3_1.max()
            cam3_2 /= cam3_2.max()
            cam3_3 /= cam3_3.max()
            cam3_sum /= cam3_sum.max()

            saveCAM_Image(inputMap=cam3_1, outdir=save_dir_sc1, fileCounter=out_counter)
            saveCAM_Image(inputMap=cam3_2, outdir=save_dir_sc2, fileCounter=out_counter)
            saveCAM_Image(inputMap=cam3_3, outdir=save_dir_sc3, fileCounter=out_counter)
            saveCAM_Image(inputMap=cam3_sum, outdir=save_dir_sum, fileCounter=out_counter)

            print('Image ' + str(out_counter) + '.png has been saved')
            out_counter += 1



with tf.variable_scope('Learning_rate'):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=lr,
                                               global_step=global_step,
                                               decay_steps=tr_iteration,
                                               decay_rate=decay_rate,
                                               staircase=True)

with tf.variable_scope('BN_switch'):
    is_training = tf.placeholder(tf.bool)

with tf.variable_scope('Input'):
    with tf.variable_scope('Input_x'):
        x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1])
    with tf.variable_scope('Input_y'):
        y_ = tf.placeholder(tf.int32, shape=[None, NUM_LABELS])

image_summary(x, layer_name='input_image', input_img=True)
tf.summary.histogram('x_image', x)

y_conv, sec1_att_input, sec1_sig_weight, netLayerSets = inference(input=x,
                                                                  ksize=ksize,
                                                                  out_channel1=128,
                                                                  out_channel2=256,
                                                                  out_channel3=512)

conv_cam, y_cam = pre_grad_cam()

with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    l2 = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = cost + l2
    tf.summary.scalar('loss', loss)

with tf.name_scope('Train_step'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, use_nesterov=True).minimize(loss,
                                                                                                              global_step)

with tf.name_scope('Accuracy'):
    y_conv_softmax = tf.nn.softmax(y_conv)
    distribution = [tf.arg_max(y_, 1), tf.arg_max(y_conv_softmax, 1)]
    correct_prediction = tf.equal(distribution[0], distribution[1])
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.variable_scope('Saver'):
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=epoch)

with tf.variable_scope('Summary_writer'):
    merged = tf.summary.merge_all()
    if Training:
        writer_training = tf.summary.FileWriter(checkpoint_dir, sess.graph)

if Load_Data:
    saver.restore(sess, checkpoint_dir + model_name + best_epoch + '.ckpt')
    print('restore finished')
else:
    sess.run(tf.global_variables_initializer())

if Training:
    best_val_accuracy = []

    batch = []
    for e in range(1, epoch + 1):
        prac_tr_img, prac_tr_lab = init.shuffle_data(tr_img, tr_lab)

        training_accuracy = 0.0
        training_loss = 0.0
        for itr in range(tr_iteration):
            img_tr, lab_tr = init.next_batch(prac_tr_img, prac_tr_lab, training_batch_size, itr)
            tr_accuracy, tr_loss, _ = sess.run([accuracy, cost, train_step], feed_dict={x: img_tr,
                                                                                        y_: lab_tr,
                                                                                        is_training: True})
            training_accuracy += tr_accuracy
            training_loss += tr_loss

        summary = sess.run(merged, feed_dict={x: img_tr,
                                              y_: lab_tr,
                                              is_training: False})

        plt_training_accuracy = float(training_accuracy / tr_iteration)
        plt_training_loss = float(training_loss / tr_iteration)

        epoch_plt.append(e)
        training_accuracy_plt.append(plt_training_accuracy)
        training_loss_plt.append(plt_training_loss)

        validation_accuracy, validation_loss = validation_procedure(loss=cost,
                                                                    val_img=val_img,
                                                                    val_lab=val_lab)
        best_val_accuracy.append(validation_accuracy)
        validation_accuracy_plt.append(validation_accuracy)
        validation_loss_plt.append(validation_loss)

        log = "Epoch [%d] , training accuracy [%g] ,Validation Accuracy: [%g] , Loss_training : [%g] , " \
              "Loss_validation: [%g] , learning_rate: [%g], time: %s" % \
              (e, plt_training_accuracy, validation_accuracy, plt_training_loss, validation_loss,
               sess.run(learning_rate), time.ctime(time.time()))

        save2file(log)

        writer_training.add_summary(summary, e)

        plot_acc_loss()

        saver.save(sess, checkpoint_dir + model_name + str(e) + '.ckpt')

        test_procedure(tst_img=tst_img, tst_lab=tst_lab)

    best_index = best_val_accuracy.index(max(best_val_accuracy))
    log1 = 'Highest Validation Accuracy : [%g], Epoch : [%g]' % (best_val_accuracy[best_index], best_index + 1)
    save2file(log1)

else:
    print('Test procedure :')
    # test_procedure(tst_img=tst_img, tst_lab=tst_lab)
    # plotROCInformation(tst_img, tst_lab)
    attentionVisualization(tst_img)
    # generate_GradCAM_Image()

sess.close()
