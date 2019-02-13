import sys

sys.path.append('../Pre_Processing/')
import matplotlib.pyplot as plt
import Init_dld as init
import numpy as np
import time
import os
import argparse
import tensorflow as tf
from resnet_v1 import resnet_v1_50
import tensorflow.contrib.slim as slim

plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument("-model")
parser.add_argument("-gpu")
parser.add_argument("-epoch", type=int, default=100)
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-best_epoch", default='0')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# --------------------hyper parameters-------------------- #
Training = False
augmentation = True

NUM_LABELS = 7
IMAGE_SIZE = 32

training_batch_size = 128
validation_batch_size = 128
test_batch_size = 128

epoch = args.epoch
lr = args.lr
model_name = args.model
best_epoch = args.best_epoch
# -------------------------------------------------------- #

pulmonary_category = {0: 'CON',
                      1: 'MUL_GGO',
                      2: 'HCM',
                      3: 'RET_GGO',
                      4: 'EMP',
                      5: 'NOD',
                      6: 'NOR'}

restore_dir = '../RestoreLib/resnet_v1_50.ckpt'
checkpoint_dir = '../Checkpoint/' + model_name + '/'
checkpoint_test_dir = '../Checkpoint/' + model_name + '/test/'

tr_img, tr_lab, val_img, val_lab = init.Load_data(init.data_dir, is_training=True)
tst_img, tst_lab = init.Load_data(init.data_dir, is_training=False)

# tr_img = np.concatenate([tr_img, tr_img, tr_img], axis=-1)
# val_img = np.concatenate([val_img, val_img, val_img], axis=-1)
# tst_img = np.concatenate([tst_img, tst_img, tst_img], axis=-1)

print(tr_img.shape)

tr_iteration = tr_img.shape[0] // training_batch_size

cfg_information = ''
cfg_information += '-' * 50 + '\n'
cfg_information += 'epoch:[%g]' % epoch + '\n'
cfg_information += 'model name:[%s]' % model_name + '\n'
cfg_information += 'learning rate:[%g]' % lr + '\n'
cfg_information += 'image_size:[%g]' % IMAGE_SIZE + '\n'
cfg_information += 'augmentation:[%s]' % str(augmentation) + '\n'
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

        y_score = sess.run(y_conv_softmax, feed_dict={x: tst_x,
                                              y_: tst_l,
                                              is_training: False})

        with open('../resnet_50_test_label.txt', 'a+') as f:
            for i in range(tst_l.shape[0]):
                tmp_str = ''
                for item in tst_l[i]:
                    tmp_str += (str(item) + ' ')
                f.write(tmp_str + '\n')

        with open('../resnet_50_y_score.txt', 'a+') as g:
            for i in range(y_score.shape[0]):
                tmp_str = ''
                for item in y_score[i]:
                    tmp_str += (str(item) + ' ')
                g.write(tmp_str + '\n')


with tf.variable_scope('BN_switch'):
    is_training = tf.placeholder(tf.bool)

with tf.variable_scope('Input'):
    with tf.variable_scope('Input_x'):
        x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1])
    with tf.variable_scope('Input_y'):
        y_ = tf.placeholder(tf.int32, shape=[None, NUM_LABELS])

y_conv, end_point = resnet_v1_50(inputs=x, num_classes=NUM_LABELS, is_training=is_training)

with tf.name_scope('Loss'):
    slim.losses.softmax_cross_entropy(logits=y_conv, onehot_labels=y_)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

with tf.name_scope('Train_step'):
    train_step = tf.train.AdamOptimizer(lr).minimize(total_loss)

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

if Training:
    variables_to_restore = slim.get_variables_to_restore(exclude=['resnet_v1_50/logits', 'resnet_v1_50/conv1'])
    init_fn = slim.assign_from_checkpoint_fn(restore_dir, variables_to_restore, ignore_missing_vars=True)
    sess.run(tf.global_variables_initializer())
    init_fn(sess)
else:
    saver = tf.train.Saver()
    saver.restore(sess, '../Checkpoint/resnet-v1-50/resnet-v1-50100.ckpt')

print('resnet-50 model parameters restore finished')

if Training:
    best_val_accuracy = []

    batch = []
    for e in range(1, epoch + 1):
        prac_tr_img, prac_tr_lab = init.shuffle_data(tr_img, tr_lab)

        training_accuracy = 0.0
        training_loss = 0.0
        for itr in range(tr_iteration):
            img_tr, lab_tr = init.next_batch(prac_tr_img, prac_tr_lab, training_batch_size, itr)
            tr_accuracy, tr_loss, _ = sess.run([accuracy, total_loss, train_step], feed_dict={x: img_tr,
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

        validation_accuracy, validation_loss = validation_procedure(loss=total_loss,
                                                                    val_img=val_img,
                                                                    val_lab=val_lab)
        best_val_accuracy.append(validation_accuracy)
        validation_accuracy_plt.append(validation_accuracy)
        validation_loss_plt.append(validation_loss)

        log = "Epoch [%d] , training accuracy [%g] ,Validation Accuracy: [%g] , Loss_training : [%g] , " \
              "Loss_validation: [%g] , time: %s" % \
              (e, plt_training_accuracy, validation_accuracy, plt_training_loss, validation_loss,
               time.ctime(time.time()))

        save2file(log)
        if Training:
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
