import sys

sys.path.append('../Pre_Processing/')
import matplotlib.pyplot as plt
import Init_dld_eigen as init
import numpy as np
import time
import os
import argparse
import tensorflow as tf
import tensorflow.contrib.layers as layers

plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument("-model")
parser.add_argument("-gpu")
parser.add_argument("-best_epoch", default='0')
parser.add_argument("-epoch", type=int, default=200)
parser.add_argument("-lr", type=float, default=0.01)
parser.add_argument("-weight_decay", type=float, default=1e-4)
parser.add_argument("-momentum", type=float, default=0.9)
parser.add_argument("-decay_rate", type=float, default=0.97)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

pulmonary_category = {0: 'CON',
                      1: 'MUL_GGO',
                      2: 'HCM',
                      3: 'RET_GGO',
                      4: 'EMP',
                      5: 'NOD',
                      6: 'NOR'}

epoch = args.epoch
start_learning_rate = args.lr
weight_decay = args.weight_decay
momentum = args.momentum
decay_rate = args.decay_rate
dropout = 0.8

training_batch_size = 128
validation_batch_size = 128
test_batch_size = 128

Training = True
Load_Data = not Training
model_name = args.model
checkpoint_dir = '../Checkpoint/' + model_name + '/'
best_epoch = args.best_epoch

tr_img, tr_eig, tr_lab, val_img, val_eig, val_lab = init.Load_data(init.data_dir, is_training=True)
tst_img, tst_eig, tst_lab = init.Load_data(init.data_dir, is_training=False)

tr_iteration = tr_img.shape[0] // training_batch_size

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

epoch_plt = []
training_accuracy_plt = []
validation_accuracy_plt = []
training_loss_plt = []
validation_loss_plt = []


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
    plt.ylim([0.0, 5.0])
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


def single_layer(input, in_channel, out_channel, conv_ksize, layer_name, relu=False, trans=False):
    with tf.variable_scope(layer_name):
        weight = tf.get_variable('weight', [conv_ksize, conv_ksize, in_channel, out_channel],
                                 initializer=layers.variance_scaling_initializer(),
                                 regularizer=layers.l2_regularizer(weight_decay))
        bias = tf.get_variable('bias', [out_channel], initializer=tf.zeros_initializer())
        if trans:
            conv = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME') + bias
        else:
            conv = tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='SAME') + bias
        tf.summary.histogram(' ', conv)
        conv = tf.layers.batch_normalization(conv, training=is_training)
        if relu:
            conv = tf.nn.relu(conv)
    return conv


def trans_dimension_image(input, in_channel, out_channel, trans_name):
    with tf.variable_scope(trans_name):
        weight = tf.get_variable('weight', [1, 1, in_channel, out_channel],
                                 initializer=layers.variance_scaling_initializer(),
                                 regularizer=layers.l2_regularizer(weight_decay))
        result = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')

    return result


def trans_dimension(input, in_channel, out_channel, trans_name):
    with tf.variable_scope(trans_name):
        weight = tf.get_variable('weight', [1, 1, in_channel, out_channel],
                                 initializer=layers.variance_scaling_initializer(),
                                 regularizer=layers.l2_regularizer(weight_decay))
        result = tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='SAME')

    return result


def block(input, in_channel, out_channel1, out_channel2, conv_ksize, Block_name, image_trans=False,
          dimention_trans=False):
    trans_input, B_layer1 = [], []
    if image_trans:
        trans_input = trans_dimension_image(input, in_channel, out_channel2, Block_name + '/trans')
        B_layer1 = single_layer(input, in_channel, out_channel1, conv_ksize, Block_name + '/layer1', relu=True,
                                trans=True)

    if not image_trans and dimention_trans:
        trans_input = trans_dimension(input, in_channel, out_channel2, Block_name + '/trans')
        B_layer1 = single_layer(input, in_channel, out_channel1, conv_ksize, Block_name + '/layer1', relu=True,
                                trans=False)

    if not image_trans and not dimention_trans:
        trans_input = input
        B_layer1 = single_layer(input, in_channel, out_channel1, conv_ksize, Block_name + '/layer1', relu=True,
                                trans=False)

    B_layer2 = single_layer(B_layer1, out_channel1, out_channel2, conv_ksize, Block_name + '/layer2', relu=False,
                            trans=False)

    result_summary = tf.add(trans_input, B_layer2)

    result_summary = tf.nn.relu(result_summary)

    return result_summary


def section(input, in_channel, out_channel1, out_channel2, conv_ksize, section_name, block_num, image_trans=False,
            dimention_trans=False):
    with tf.variable_scope(section_name):
        block_result = block(input, in_channel, out_channel1, out_channel2, conv_ksize, 'block1',
                             image_trans=image_trans, dimention_trans=dimention_trans)
        for i in range(block_num - 1):
            block_result = block(block_result, out_channel2, out_channel1, out_channel2, conv_ksize,
                                 'block' + str(i + 2), image_trans=False, dimention_trans=False)
        block_result = tf.nn.dropout(block_result, keep_prob=keep_prob)
    return block_result


def f_value(matrix):
    f = 0.0
    length = len(matrix[0])
    for i in range(length):
        recall = matrix[i][i] / np.sum([matrix[i][m] for m in range(7)])
        precision = matrix[i][i] / np.sum([matrix[n][i] for n in range(7)])
        result = (recall * precision) / (recall + precision)
        f += result
    f *= (2 / 7)
    return f


def validation_procedure(loss, val_img, val_eig, val_lab):
    confusion_matrics = np.zeros([7, 7], dtype="int")
    val_loss = 0.0

    val_batch_num = int(np.ceil(val_img.shape[0] / validation_batch_size))
    for step in range(val_batch_num):
        val_x = val_img[step * validation_batch_size:step * validation_batch_size + validation_batch_size]
        val_e = val_eig[step * validation_batch_size:step * validation_batch_size + validation_batch_size]
        val_l = val_lab[step * validation_batch_size:step * validation_batch_size + validation_batch_size]

        [matrix_row, matrix_col], tmp_loss = sess.run([distribution, loss],
                                                      feed_dict={x_normal: val_x,
                                                                 x_eigen: val_e,
                                                                 y_: val_l,
                                                                 keep_prob: 1.0,
                                                                 is_training: False})
        for m, n in zip(matrix_row, matrix_col):
            confusion_matrics[m][n] += 1

        val_loss += tmp_loss

    validation_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(7)])) / float(
        np.sum(confusion_matrics))
    validation_loss = val_loss / val_batch_num

    return validation_accuracy, validation_loss


def plotROCInformation(tst_img, tst_eig, tst_lab):
    tst_batch_num = int(np.ceil(tst_img.shape[0] / test_batch_size))

    for step in range(tst_batch_num):
        print('Tst_batch_num' + str(tst_batch_num))
        tst_x = tst_img[step * test_batch_size:step * test_batch_size + test_batch_size]
        tst_e = tst_eig[step * test_batch_size:step * test_batch_size + test_batch_size]
        tst_l = tst_lab[step * test_batch_size:step * test_batch_size + test_batch_size]

        y_score = sess.run(y_conv, feed_dict={x_normal: tst_x,
                                              x_eigen: tst_e,
                                              y_: tst_l,
                                              keep_prob: 1.0,
                                              is_training: False})

        with open('../db_resnet_test_label.txt', 'a+') as f:
            for i in range(tst_l.shape[0]):
                tmp_str = ''
                for item in tst_l[i]:
                    tmp_str += (str(item) + ' ')
                f.write(tmp_str + '\n')

        with open('../db_resnet_y_score.txt', 'a+') as g:
            for i in range(y_score.shape[0]):
                tmp_str = ''
                for item in y_score[i]:
                    tmp_str += (str(item) + ' ')
                g.write(tmp_str + '\n')


def test_procedure(tst_img, tst_eig, tst_lab):
    confusion_matrics = np.zeros([7, 7], dtype="int")

    tst_batch_num = int(np.ceil(tst_img.shape[0] / test_batch_size))
    for step in range(tst_batch_num):
        print('Tst_batch_num' + str(tst_batch_num))
        tst_x = tst_img[step * test_batch_size:step * test_batch_size + test_batch_size]
        tst_e = tst_eig[step * test_batch_size:step * test_batch_size + test_batch_size]
        tst_l = tst_lab[step * test_batch_size:step * test_batch_size + test_batch_size]

        matrix_row, matrix_col = sess.run(distribution,
                                          feed_dict={x_normal: tst_x,
                                                     x_eigen: tst_e,
                                                     y_: tst_l,
                                                     keep_prob: 1.0,
                                                     is_training: False})
        for m, n in zip(matrix_row, matrix_col):
            confusion_matrics[m][n] += 1

    test_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(7)])) / float(
        np.sum(confusion_matrics))
    detail_test_accuracy = [confusion_matrics[i][i] / np.sum(confusion_matrics[i]) for i in range(7)]
    log1 = "Test Accuracy : %g" % test_accuracy
    log2 = np.array(confusion_matrics.tolist())
    log3 = ''
    for j in range(7):
        log3 += 'category %s test accuracy : %g\n' % (pulmonary_category[j], detail_test_accuracy[j])
    log4 = 'F_Value : %g' % f_value(confusion_matrics)

    save2file(log1)
    save2file(log2)
    save2file(log3)
    save2file(log4)


with tf.name_scope('Input'):
    with tf.name_scope('Input_x'):
        x_normal = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
        x_eigen = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    with tf.name_scope('Input_y'):
        y_ = tf.placeholder(tf.int32, shape=[None, 7])
    with tf.name_scope('Image_summary'):
        tf.summary.image('x_normal', x_normal, max_outputs=3)
        tf.summary.image('x_eigen', x_eigen, max_outputs=3)

with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('Bn_switch'):
    is_training = tf.placeholder(tf.bool)

# 32*32
res_normal_input = single_layer(input=x_normal, in_channel=1, out_channel=64, conv_ksize=2,
                                layer_name='res_normal_input', relu=True, trans=False)
res_normal_maxpool = tf.nn.max_pool(res_normal_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
res_eigen_input = single_layer(input=x_eigen, in_channel=3, out_channel=64, conv_ksize=2,
                               layer_name='res_eigen_input', relu=True, trans=False)
res_eigen_maxpool = tf.nn.max_pool(res_eigen_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 16*16*64
section1_normal = section(input=res_normal_maxpool,
                          in_channel=res_normal_maxpool.get_shape()[3].value,
                          out_channel1=64,
                          out_channel2=64,
                          conv_ksize=2,
                          section_name='Normal_Section_1',
                          block_num=2,
                          image_trans=False,
                          dimention_trans=False)

section1_eigen = section(input=res_eigen_maxpool,
                         in_channel=res_eigen_maxpool.get_shape()[3].value,
                         out_channel1=64,
                         out_channel2=64,
                         conv_ksize=2,
                         section_name='eigen_Section_1',
                         block_num=2,
                         image_trans=False,
                         dimention_trans=False)

# 16*16*64
section2_normal = section(input=section1_normal,
                          in_channel=section1_normal.get_shape()[3].value,
                          out_channel1=128,
                          out_channel2=128,
                          conv_ksize=2,
                          section_name='Normal_Section_2',
                          block_num=2,
                          image_trans=True,
                          dimention_trans=True)

section2_eigen = section(input=section1_eigen,
                         in_channel=section1_eigen.get_shape()[3].value,
                         out_channel1=128,
                         out_channel2=128,
                         conv_ksize=2,
                         section_name='eigen_Section_2',
                         block_num=2,
                         image_trans=True,
                         dimention_trans=True)

# 8*8*128
section3_normal = section(input=section2_normal,
                          in_channel=section2_normal.get_shape()[3].value,
                          out_channel1=256,
                          out_channel2=256,
                          conv_ksize=2,
                          section_name='Normal_Section_3',
                          block_num=2,
                          image_trans=True,
                          dimention_trans=True)

section3_eigen = section(input=section2_eigen,
                         in_channel=section2_eigen.get_shape()[3].value,
                         out_channel1=256,
                         out_channel2=256,
                         conv_ksize=2,
                         section_name='eigen_Section_3',
                         block_num=2,
                         image_trans=True,
                         dimention_trans=True)

# 4*4*256
section4_normal = section(input=section3_normal,
                          in_channel=section3_normal.get_shape()[3].value,
                          out_channel1=512,
                          out_channel2=512,
                          conv_ksize=2,
                          section_name='Normal_Section_4',
                          block_num=1,
                          image_trans=True,
                          dimention_trans=True)

section4_eigen = section(input=section3_eigen,
                         in_channel=section3_eigen.get_shape()[3].value,
                         out_channel1=512,
                         out_channel2=512,
                         conv_ksize=2,
                         section_name='eigen_Section_4',
                         block_num=1,
                         image_trans=True,
                         dimention_trans=True)

# 2*2*512
max_pool_normal = tf.nn.max_pool(section4_normal, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
max_pool_eigen = tf.nn.max_pool(section4_eigen, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sum_avg = tf.concat([max_pool_normal, max_pool_eigen], axis=3)

with tf.variable_scope('Full_Connected_Layer'):
    sum_avg_flat = tf.reshape(sum_avg, [-1, int(sum_avg.get_shape()[3])])
    weight = tf.get_variable('weight', [sum_avg_flat.get_shape()[1], 7],
                             initializer=layers.variance_scaling_initializer(),
                             regularizer=layers.l2_regularizer(weight_decay))
    bias = tf.get_variable('bias', [7], initializer=tf.zeros_initializer())
    y_conv = tf.nn.bias_add(tf.matmul(sum_avg_flat, weight), bias)
    tf.summary.histogram(' ', y_conv)

with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    l2 = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = cost + l2
    tf.summary.scalar('Loss', loss)

with tf.variable_scope('Learning_rate'):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=start_learning_rate,
                                               global_step=global_step,
                                               decay_steps=tr_iteration,
                                               decay_rate=decay_rate,
                                               staircase=True)

with tf.name_scope('Train_step'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, use_nesterov=True).minimize(loss,
                                                                                                              global_step)

with tf.name_scope('Accuracy'):
    y_conv_softmax = tf.nn.softmax(y_conv)
    distribution = [tf.arg_max(y_, 1), tf.arg_max(y_conv, 1)]
    correct_prediction = tf.equal(distribution[0], distribution[1])

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('Accuracy', accuracy)

with tf.variable_scope('Saver'):
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=epoch)

merged = tf.summary.merge_all()
writer_training = tf.summary.FileWriter(checkpoint_dir + 'train/', sess.graph)

if Load_Data:
    saver.restore(sess, checkpoint_dir + model_name + best_epoch + '.ckpt')
else:
    sess.run(tf.global_variables_initializer())

if Training:
    best_val_accuracy = []

    batch = []
    for e in range(1, epoch + 1):
        prac_tr_img, prac_tr_eigen, prac_tr_lab = init.shuffle_data(tr_img, tr_eig, tr_lab)

        training_accuracy = 0.0
        training_loss = 0.0
        for itr in range(tr_iteration):
            img_tr, eig_tr, lab_tr = init.next_batch(prac_tr_img, prac_tr_eigen, prac_tr_lab, training_batch_size, itr)
            tr_accuracy, tr_loss, _ = sess.run([accuracy, cost, train_step], feed_dict={x_normal: img_tr,
                                                                                        x_eigen: eig_tr,
                                                                                        y_: lab_tr,
                                                                                        keep_prob: dropout,
                                                                                        is_training: True})
            training_accuracy += tr_accuracy
            training_loss += tr_loss

        summary = sess.run(merged, feed_dict={x_normal: img_tr,
                                              x_eigen: eig_tr,
                                              y_: lab_tr,
                                              keep_prob: 1.0,
                                              is_training: False})

        plt_training_accuracy = float(training_accuracy / tr_iteration)
        plt_training_loss = float(training_loss / tr_iteration)

        epoch_plt.append(e)
        training_accuracy_plt.append(plt_training_accuracy)
        training_loss_plt.append(plt_training_loss)

        validation_accuracy, validation_loss = validation_procedure(loss=cost,
                                                                    val_img=val_img,
                                                                    val_eig=val_eig,
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

        test_procedure(tst_img=tst_img, tst_eig=tst_eig, tst_lab=tst_lab)

    best_index = best_val_accuracy.index(max(best_val_accuracy))
    log1 = 'Highest Validation Accuracy : [%g], Epoch : [%g]' % (best_val_accuracy[best_index], best_index + 1)
    save2file(log1)

else:
    print('Test procedure :')
    test_procedure(tst_img=tst_img, tst_eig=tst_eig, tst_lab=tst_lab)
    # plotROCInformation(tst_img, tst_eig, tst_lab)

sess.close()
