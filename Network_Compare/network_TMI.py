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
parser.add_argument("-best_epoch", default='0')
parser.add_argument("-epoch", type=int, default=200)
parser.add_argument("-lr", type=float, default=1e-3)
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
k = 4
dropout = 0.5
leaky_alpha = 0.3

training_batch_size = 128
validation_batch_size = 128
test_batch_size = 128

Training = True
Load_Data = not Training

model_name = args.model
checkpoint_dir = '../Checkpoint/' + model_name + '/'
best_epoch = args.best_epoch

tr_img, tr_lab, val_img, val_lab = init.Load_data(init.data_dir, is_training=True)
tst_img, tst_lab = init.Load_data(init.data_dir, is_training=False)

tr_iteration = tr_img.shape[0] // training_batch_size

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


def conv_layer(input, conv_ksize, in_channel, out_channel, layer_name):
    with tf.variable_scope(layer_name):
        weight = tf.get_variable('weight', shape=[conv_ksize, conv_ksize, in_channel, out_channel],
                                 initializer=tf.orthogonal_initializer(gain=1.1))
        bias = tf.get_variable('bias', shape=[out_channel], initializer=tf.zeros_initializer())
        conv = tf.nn.bias_add(tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='VALID'), bias)
        conv = leaky_relu(conv)
        tf.summary.histogram(' ', conv)
    return conv


def leaky_relu(x, alpha=leaky_alpha):
    with tf.variable_scope('Leaky_Relu'):
        return tf.nn.leaky_relu(x, alpha=alpha)


def avg_pool(x, size):
    with tf.variable_scope('Average_Pooling'):
        return tf.nn.avg_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')


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
                                                                 y_: val_l,
                                                                 keep_prob: 1.0})
        for m, n in zip(matrix_row, matrix_col):
            confusion_matrics[m][n] += 1

        val_loss += tmp_loss

    validation_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(7)])) / float(
        np.sum(confusion_matrics))
    validation_loss = val_loss / val_batch_num

    return validation_accuracy, validation_loss


# def plotROCInformation(tst_img, tst_lab):
#     tst_batch_num = int(np.ceil(tst_img.shape[0] / test_batch_size))
#
#     for step in range(tst_batch_num):
#         print('Tst_batch_num' + str(tst_batch_num))
#         tst_x = tst_img[step * test_batch_size:step * test_batch_size + test_batch_size]
#         tst_l = tst_lab[step * test_batch_size:step * test_batch_size + test_batch_size]
#
#         y_score = sess.run(y_conv, feed_dict={x: tst_x,
#                                               y_: tst_l,
#                                               keep_prob: 1.0})
#
#         with open('../tmi_test_label.txt', 'a+') as f:
#             for i in range(tst_l.shape[0]):
#                 tmp_str = ''
#                 for item in tst_l[i]:
#                     tmp_str += (str(item) + ' ')
#                 f.write(tmp_str + '\n')
#
#         with open('../tmi_y_score.txt', 'a+') as g:
#             for i in range(y_score.shape[0]):
#                 tmp_str = ''
#                 for item in y_score[i]:
#                     tmp_str += (str(item) + ' ')
#                 g.write(tmp_str + '\n')


def test_procedure(tst_img, tst_lab):
    confusion_matrics = np.zeros([7, 7], dtype="int")

    tst_batch_num = int(np.ceil(tst_img.shape[0] / test_batch_size))
    for step in range(tst_batch_num):
        print('Tst_batch_num' + str(tst_batch_num))
        tst_x = tst_img[step * test_batch_size:step * test_batch_size + test_batch_size]
        tst_l = tst_lab[step * test_batch_size:step * test_batch_size + test_batch_size]

        matrix_row, matrix_col = sess.run(distribution,
                                          feed_dict={x: tst_x,
                                                     y_: tst_l,
                                                     keep_prob: 1.0})
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


# def pre_grad_cam(conv_op, fc_score_op):
#     predicted_class = tf.arg_max(tf.nn.softmax(y_conv), 1)
#
#     conv_layer_cam1 = conv_op
#
#     y_score_cam1 = fc_score_op
#
#     one_hot = tf.one_hot(indices=predicted_class, depth=7)
#     signal_cam1 = tf.multiply(y_score_cam1, one_hot)
#
#     loss_cam1 = tf.reduce_mean(signal_cam1)
#
#     grads_cam1 = tf.gradients(loss_cam1, conv_layer_cam1)[0]
#
#     norm_grads_cam1 = tf.div(grads_cam1, tf.sqrt(tf.reduce_mean(tf.square(grads_cam1))) + tf.constant(1e-5))
#
#     return conv_layer_cam1, norm_grads_cam1


# def grad_cam(input, conv_cam, norm_gradient):
#     output, grads_val = sess.run([conv_cam, norm_gradient], feed_dict={x: input, keep_prob: 1.0})
#     output = output[0]
#     grads_val = grads_val[0]
#
#     weights = np.mean(grads_val, axis=(0, 1))
#     cam = np.ones(output.shape[0:2], dtype=np.float32)
#
#     for i, w in enumerate(weights):
#         cam += w * output[:, :, i]
#
#     cam = np.maximum(cam, 0)
#     cam = cv2.resize(cam, (32, 32), interpolation=cv2.INTER_CUBIC)
#     cam = cam / np.max(cam)
#
#     return cam
#
# def saveCAM_Image(inputMap, outdir, fileCounter):
#     fig, ax = plt.subplots()
#     ax.imshow(inputMap, cmap=plt.cm.jet, alpha=1.0, interpolation='nearest', vmin=0, vmax=1)
#     plt.axis('off')
#
#     height, width = inputMap.shape
#
#     fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
#     plt.margins(0, 0)
#
#     plt.savefig(outdir + str(fileCounter) + '.png', dpi=300)
#     plt.close()
#
#
# def generate_GradCAM_Image(save_dir='../Grad_CAM_TMI_Split/'):
#     save_dir_sc1 = save_dir
#
#     if not os.path.isdir(save_dir_sc1):
#         os.makedirs(save_dir_sc1)
#
#     tst_batch_num = int(np.ceil(tst_img.shape[0] / test_batch_size))
#
#     out_counter = 1
#
#     for step in range(tst_batch_num):
#         print('step : ' + str(step))
#         tst_x = tst_img[step * test_batch_size:step * test_batch_size + test_batch_size]
#
#         for m in range(0, tst_x.shape[0], 8):
#             imgForCal = np.expand_dims(tst_x[m], 0)
#
#             cam3_1 = grad_cam(imgForCal, conv_cam, y_cam)
#
#             cam3_1 /= cam3_1.max()
#
#             saveCAM_Image(inputMap=cam3_1, outdir=save_dir_sc1, fileCounter=out_counter)
#
#             out_counter += 8
#             print('Image ' + str(out_counter) + '.png has been saved')


with tf.name_scope('Input'):
    with tf.name_scope('Input_x'):
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
    with tf.name_scope('Input_y'):
        y_ = tf.placeholder(tf.int32, shape=[None, 7])

with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)

conv = conv_layer(input=x, conv_ksize=2, in_channel=1, out_channel=4 * k, layer_name='conv1')
conv = conv_layer(input=conv, conv_ksize=2, in_channel=4 * k, out_channel=9 * k, layer_name='conv2')
conv = conv_layer(input=conv, conv_ksize=2, in_channel=9 * k, out_channel=16 * k, layer_name='conv3')
conv = conv_layer(input=conv, conv_ksize=2, in_channel=16 * k, out_channel=25 * k, layer_name='conv4')
conv = conv_layer(input=conv, conv_ksize=2, in_channel=25 * k, out_channel=36 * k, layer_name='conv5')

with tf.variable_scope('Trans_Dropout'):
    avg_conv = avg_pool(conv, conv.get_shape()[1].value)
    avg_conv_reshape = tf.reshape(avg_conv, [-1, 1 * 1 * 36 * k])
    avg_conv_reshape_dropout = tf.nn.dropout(avg_conv_reshape, keep_prob=keep_prob)

with tf.variable_scope('FC1'):
    weight = tf.get_variable('weight', shape=[avg_conv_reshape_dropout.get_shape()[1].value, 6 * 36 * k],
                             initializer=layers.variance_scaling_initializer())
    bias = tf.get_variable('bias', shape=[6 * 36 * k],
                           initializer=tf.zeros_initializer())
    fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(avg_conv_reshape_dropout, weight), bias))
    tf.summary.histogram(' ', fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

with tf.variable_scope('FC2'):
    weight = tf.get_variable('weight', shape=[6 * 36 * k, 2 * 36 * k],
                             initializer=layers.variance_scaling_initializer())
    bias = tf.get_variable('bias', shape=[2 * 36 * k],
                           initializer=tf.zeros_initializer())
    fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1, weight), bias))
    tf.summary.histogram(' ', fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)

with tf.variable_scope('FC3'):
    weight = tf.get_variable('weight', shape=[2 * 36 * k, 7],
                             initializer=layers.variance_scaling_initializer())
    bias = tf.get_variable('bias', shape=[7],
                           initializer=tf.zeros_initializer())
    y_conv = tf.nn.bias_add(tf.matmul(fc2, weight), bias)
    tf.summary.histogram(' ', y_conv)

    # conv_cam, y_cam = pre_grad_cam(conv, y_conv)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    tf.summary.scalar('Loss', loss)

with tf.name_scope('Train_step'):
    train_step = tf.train.AdamOptimizer(start_learning_rate).minimize(loss)

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
                                                                                        y_: lab_tr,
                                                                                        keep_prob: dropout})
            training_accuracy += tr_accuracy
            training_loss += tr_loss

        summary = sess.run(merged, feed_dict={x: img_tr,
                                              y_: lab_tr,
                                              keep_prob: 1.0})

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
               start_learning_rate, time.ctime(time.time()))

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
    # plotROCInformation(tst_img, tst_lab)
    # generate_GradCAM_Image()

sess.close()
