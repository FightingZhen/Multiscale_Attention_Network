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

plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument("-model")
parser.add_argument("-gpu")
parser.add_argument("-epoch", type=int, default=200)
parser.add_argument("-lr", type=float, default=0.01)
parser.add_argument("-weight_decay", type=float, default=1e-3)
parser.add_argument("-momentum", type=float, default=0.9)
parser.add_argument("-best_epoch", default='0')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# --------------------hyper parameters-------------------- #
Training = True
augmentation = True

NUM_LABELS = 7
IMAGE_SIZE = 32

training_batch_size = 128
validation_batch_size = 128
test_batch_size = 128

epoch = args.epoch
momentum = args.momentum
weight_decay = args.weight_decay
lr = args.lr
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
cfg_information += 'Image_size:[%g]' % IMAGE_SIZE + '\n'
cfg_information += 'Augmentation:[%s]' % str(augmentation) + '\n'
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


def conv_layer(input, out_channel, ksize, stride, std, layer_name):
    with tf.variable_scope(layer_name):
        weight = tf.get_variable('weight', [ksize, ksize, input.get_shape()[-1], out_channel],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=std),
                                 regularizer=layers.l2_regularizer(weight_decay))
        result = tf.nn.conv2d(input, weight, strides=[1, stride, stride, 1], padding='SAME')

        tf.summary.histogram('weight', weight)
        tf.summary.histogram('result', result)

    return result


def relu_layer(input, layer_name):
    with tf.variable_scope(layer_name):
        return tf.nn.relu(input)


def lrn_layer(input, layer_name):
    with tf.variable_scope(layer_name):
        return tf.nn.lrn(input)


def dropoutLayer(input, keep_prob, layer_name):
    with tf.variable_scope(layer_name):
        return tf.nn.dropout(input, keep_prob=keep_prob)


def max_pool_layer(input, ksize, stride, layer_name):
    with tf.variable_scope(layer_name):
        return tf.nn.max_pool(input,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')


def flatten(input, layer_name):
    with tf.variable_scope(layer_name):
        return tf.layers.flatten(input)


def fc_layer(input, out_channel, std, layer_name):
    with tf.variable_scope(layer_name):
        in_channel = input.get_shape().as_list()[-1]
        weight = tf.get_variable('weight', [in_channel, out_channel],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=std),
                                 regularizer=layers.l2_regularizer(weight_decay))

        bias = tf.get_variable('bias', [out_channel], initializer=tf.zeros_initializer())
        fc = tf.matmul(input, weight) + bias

        tf.summary.histogram('weight', weight)
        tf.summary.histogram('bias', bias)
        tf.summary.histogram('fc_result', fc)

    return fc


def inference(input):
    conv_layer1 = conv_layer(input, out_channel=64, ksize=4, stride=1, std=0.001, layer_name='conv_layer1')
    relu_layer1 = relu_layer(conv_layer1, layer_name='relu_layer1')
    lrn_layer1 = lrn_layer(relu_layer1, layer_name='lrn_layer1')
    pool_layer1 = max_pool_layer(lrn_layer1, ksize=3, stride=2, layer_name='pool_layer1')

    conv_layer2 = conv_layer(pool_layer1, out_channel=64, ksize=3, stride=1, std=0.01, layer_name='conv_layer2')
    relu_layer2 = relu_layer(conv_layer2, layer_name='relu_layer2')
    lrn_layer2 = lrn_layer(relu_layer2, layer_name='lrn_layer2')
    pool_layer2 = max_pool_layer(lrn_layer2, ksize=3, stride=2, layer_name='pool_layer2')

    conv_layer3 = conv_layer(pool_layer2, out_channel=64, ksize=3, stride=1, std=0.01, layer_name='conv_layer3')
    relu_layer3 = relu_layer(conv_layer3, layer_name='relu_layer3')

    conv_layer4 = conv_layer(relu_layer3, out_channel=64, ksize=3, stride=1, std=0.01, layer_name='conv_layer4')
    relu_layer4 = relu_layer(conv_layer4, layer_name='relu_layer4')

    flatten_layer = flatten(relu_layer4, layer_name='flatten_layer')

    fc1 = fc_layer(flatten_layer, out_channel=100, std=0.01, layer_name='fc1')
    relu_layer5 = relu_layer(fc1, layer_name='relu_layer5')
    dp_layer = dropoutLayer(relu_layer5, keep_prob=keep_prob, layer_name='dp_layer')

    fc2 = fc_layer(dp_layer, out_channel=NUM_LABELS, std=0.01, layer_name='fc2')

    return fc2


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
                                                                 is_training: False,
                                                                 keep_prob: 1.0})
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
                                                     is_training: False,
                                                     keep_prob: 1.0})
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


with tf.variable_scope('Learning_rate'):
    learning_rate = tf.Variable(lr, trainable=False)

with tf.variable_scope('BN_switch'):
    is_training = tf.placeholder(tf.bool)

with tf.variable_scope('Input'):
    with tf.variable_scope('Input_x'):
        x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1])
    with tf.variable_scope('Input_y'):
        y_ = tf.placeholder(tf.int32, shape=[None, NUM_LABELS])
    with tf.variable_scope('Keep_prob'):
        keep_prob = tf.placeholder(tf.float32)

y_conv = inference(input=x)

with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    l2 = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = cost + l2
    tf.summary.scalar('loss', loss)

with tf.name_scope('Train_step'):
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum=momentum).minimize(loss)

with tf.name_scope('Accuracy'):
    y_conv_softmax = tf.nn.softmax(y_conv)
    distribution = [tf.arg_max(y_, 1), tf.arg_max(y_conv_softmax, 1)]
    correct_prediction = tf.equal(distribution[0], distribution[1])
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.variable_scope('Saver'):
    saver = tf.train.Saver(max_to_keep=epoch)

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
                                                                                        is_training: True,
                                                                                        keep_prob: 0.5})
            training_accuracy += tr_accuracy
            training_loss += tr_loss

        summary = sess.run(merged, feed_dict={x: img_tr,
                                              y_: lab_tr,
                                              is_training: False,
                                              keep_prob: 1.0})

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
    test_procedure(tst_img=tst_img, tst_lab=tst_lab)

sess.close()
