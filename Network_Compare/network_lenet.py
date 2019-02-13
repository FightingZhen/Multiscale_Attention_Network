import sys

sys.path.append('../Pre_Processing/')
import matplotlib.pyplot as plt
import Init_dld as init
import numpy as np
import time
import os
import argparse
import tensorflow as tf

plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument("-model")
parser.add_argument("-gpu")
parser.add_argument("-best_epoch", default='0')
parser.add_argument("-epoch", type=int, default=100)
parser.add_argument("-lr", type=float, default=1e-4)
args = parser.parse_args()

epoch_plt = []
training_accuracy_plt = []
validation_accuracy_plt = []
training_loss_plt = []
validation_loss_plt = []

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
stddev = 0.1

training_batch_size = 32
validation_batch_size = 128
test_batch_size = 128

Training = False
Load_Data = not Training

model_name = args.model
checkpoint_dir = '../Checkpoint/' + model_name + '/'
best_epoch = args.best_epoch

tr_img, tr_lab, val_img, val_lab = init.Load_data(init.data_dir, is_training=True)
tst_img, tst_lab = init.Load_data(init.data_dir, is_training=False)

tr_iteration = tr_img.shape[0] // training_batch_size

sess = tf.Session()


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


def conv_layer(input, conv_ksize, in_channel, out_channel, layer_name):
    with tf.variable_scope(layer_name):
        weight = tf.get_variable('weight', shape=[conv_ksize, conv_ksize, in_channel, out_channel],
                                 initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', shape=[out_channel], initializer=tf.zeros_initializer())
        conv = tf.nn.bias_add(tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='VALID'), bias)
        conv = tf.nn.relu(conv)
        tf.summary.histogram(' ', conv)
    return conv


def max_pool(x, size):
    with tf.variable_scope('Max_Pooling'):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')


def save2file(log_info):
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    logfile = open(checkpoint_dir + model_name + '.txt', 'a+')
    print(log_info)
    print(log_info, file=logfile)
    logfile.close()


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


def validation_procedure(loss, val_img, val_lab):
    confusion_matrics = np.zeros([7, 7], dtype="int")
    val_loss = 0.0

    val_batch_num = int(np.ceil(val_img.shape[0] / validation_batch_size))
    for step in range(val_batch_num):
        val_x = val_img[step * validation_batch_size:step * validation_batch_size + validation_batch_size]
        val_l = val_lab[step * validation_batch_size:step * validation_batch_size + validation_batch_size]

        [matrix_row, matrix_col], tmp_loss = sess.run([distribution, loss],
                                                      feed_dict={x: val_x,
                                                                 y_: val_l})
        for m, n in zip(matrix_row, matrix_col):
            confusion_matrics[m][n] += 1

        val_loss += tmp_loss

    validation_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(7)])) / float(
        np.sum(confusion_matrics))
    validation_loss = val_loss / val_batch_num

    return validation_accuracy, validation_loss


def plotROCInformation(tst_img, tst_lab):
    tst_batch_num = int(np.ceil(tst_img.shape[0] / test_batch_size))

    for step in range(tst_batch_num):
        print('Tst_batch_num' + str(tst_batch_num))
        tst_x = tst_img[step * test_batch_size:step * test_batch_size + test_batch_size]
        tst_l = tst_lab[step * test_batch_size:step * test_batch_size + test_batch_size]

        y_score = sess.run(y_conv, feed_dict={x: tst_x,
                                              y_: tst_l})

        with open('../lenet_test_label.txt', 'a+') as f:
            for i in range(tst_l.shape[0]):
                tmp_str = ''
                for item in tst_l[i]:
                    tmp_str += (str(item) + ' ')
                f.write(tmp_str + '\n')

        with open('../lenet_y_score.txt', 'a+') as g:
            for i in range(y_score.shape[0]):
                tmp_str = ''
                for item in y_score[i]:
                    tmp_str += (str(item) + ' ')
                g.write(tmp_str + '\n')


def test_procedure(tst_img, tst_lab):
    confusion_matrics = np.zeros([7, 7], dtype="int")

    tst_batch_num = int(np.ceil(tst_img.shape[0] / test_batch_size))
    for step in range(tst_batch_num):
        print('Tst_batch_num' + str(tst_batch_num))
        tst_x = tst_img[step * test_batch_size:step * test_batch_size + test_batch_size]
        tst_l = tst_lab[step * test_batch_size:step * test_batch_size + test_batch_size]

        matrix_row, matrix_col = sess.run(distribution,
                                          feed_dict={x: tst_x,
                                                     y_: tst_l})
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
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
    with tf.name_scope('Input_y'):
        y_ = tf.placeholder(tf.int32, shape=[None, 7])

conv = conv_layer(input=x, conv_ksize=5, in_channel=1, out_channel=6, layer_name='conv1')
conv = max_pool(x=conv, size=2)
conv = conv_layer(input=conv, conv_ksize=5, in_channel=6, out_channel=16, layer_name='conv2')
conv = max_pool(x=conv, size=2)
conv = conv_layer(input=conv, conv_ksize=5, in_channel=16, out_channel=120, layer_name='conv3')

with tf.variable_scope('Reshape'):
    conv_reshape = tf.reshape(conv, [-1, 1 * 1 * 120])

with tf.variable_scope('FC1'):
    weight = tf.get_variable('weight', shape=[120, 84],
                             initializer=tf.truncated_normal_initializer(stddev=stddev))
    bias = tf.get_variable('bias', shape=[84], initializer=tf.zeros_initializer())
    fc1 = tf.nn.bias_add(tf.matmul(conv_reshape, weight), bias)
    tf.summary.histogram(' ', fc1)

with tf.variable_scope('FC2'):
    weight = tf.get_variable('weight', shape=[84, 7],
                             initializer=tf.truncated_normal_initializer(stddev=stddev))
    bias = tf.get_variable('bias', shape=[7], initializer=tf.zeros_initializer())
    y_conv = tf.nn.bias_add(tf.matmul(fc1, weight), bias)
    tf.summary.histogram(' ', y_conv)

with tf.variable_scope('Learning_rate'):
    learning_rate = tf.Variable(start_learning_rate, trainable=False)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    tf.summary.scalar('Loss', loss)

with tf.name_scope('Train_step'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('Accuracy'):
    y_conv_softmax = tf.nn.softmax(y_conv)
    distribution = [tf.arg_max(y_, 1), tf.arg_max(y_conv, 1)]
    correct_prediction = tf.equal(distribution[0], distribution[1])

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('Accuracy', accuracy)

saver = tf.train.Saver(max_to_keep=epoch)

merged = tf.summary.merge_all()
writer_training = tf.summary.FileWriter(checkpoint_dir + 'train/', sess.graph)
writer_validation = tf.summary.FileWriter(checkpoint_dir + 'validation/', sess.graph)

if Load_Data:
    saver.restore(sess, checkpoint_dir + model_name + best_epoch + '.ckpt')
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
            tr_accuracy, tr_loss, _ = sess.run([accuracy, loss, train_step], feed_dict={x: img_tr,
                                                                                        y_: lab_tr})
            training_accuracy += tr_accuracy
            training_loss += tr_loss

        summary = sess.run(merged, feed_dict={x: img_tr,
                                              y_: lab_tr})

        plt_training_accuracy = float(training_accuracy / tr_iteration)
        plt_training_loss = float(training_loss / tr_iteration)

        epoch_plt.append(e)
        training_accuracy_plt.append(plt_training_accuracy)
        training_loss_plt.append(plt_training_loss)

        validation_accuracy, validation_loss = validation_procedure(loss=loss,
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
    plotROCInformation(tst_img, tst_lab)

sess.close()
