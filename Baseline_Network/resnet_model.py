import sys

sys.path.append('../Data_Initialization/')
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import Init_dld as init
import utils
import time


class ResNet(object):
    def __init__(self, model_name, sess, train_data, val_data, tst_data, epoch, restore_epoch, num_class, ksize,
                 out_channel1, out_channel2, out_channel3, learning_rate, weight_decay, batch_size, img_height,
                 img_width, train_phase):

        self.sess = sess
        self.training_data = train_data
        self.validation_data = val_data
        self.test_data = tst_data
        self.eps = epoch
        self.res_eps = restore_epoch
        self.model = model_name
        self.ckptDir = '../checkpoint/' + self.model + '/'
        self.k = ksize
        self.oc1 = out_channel1
        self.oc2 = out_channel2
        self.oc3 = out_channel3
        self.lr = learning_rate
        self.wd = weight_decay
        self.bs = batch_size
        self.img_h = img_height
        self.img_w = img_width
        self.num_class = num_class
        self.train_phase = train_phase
        self.plt_epoch = []
        self.plt_training_accuracy = []
        self.plt_validation_accuracy = []
        self.plt_training_loss = []
        self.plt_validation_loss = []

        self.build_model()
        if self.train_phase == 'Train':
            self.saveConfiguration()

    def saveConfiguration(self):
        utils.save2file('epoch : %d' % self.eps, self.ckptDir, self.model)
        utils.save2file('restore epoch : %d' % self.res_eps, self.ckptDir, self.model)
        utils.save2file('model : %s' % self.model, self.ckptDir, self.model)
        utils.save2file('ksize : %d' % self.k, self.ckptDir, self.model)
        utils.save2file('out channel 1 : %d' % self.oc1, self.ckptDir, self.model)
        utils.save2file('out channel 2 : %d' % self.oc2, self.ckptDir, self.model)
        utils.save2file('out channel 3 : %d' % self.oc3, self.ckptDir, self.model)
        utils.save2file('learning rate : %g' % self.lr, self.ckptDir, self.model)
        utils.save2file('weight decay : %g' % self.wd, self.ckptDir, self.model)
        utils.save2file('batch size : %d' % self.bs, self.ckptDir, self.model)
        utils.save2file('image height : %d' % self.img_h, self.ckptDir, self.model)
        utils.save2file('image width : %d' % self.img_w, self.ckptDir, self.model)
        utils.save2file('num class : %d' % self.num_class, self.ckptDir, self.model)
        utils.save2file('train phase : %s' % self.train_phase, self.ckptDir, self.model)

    def convLayer(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_weight = tf.get_variable('conv_weight', [ksize, ksize, inputMap.get_shape()[-1], out_channel],
                                          initializer=layers.variance_scaling_initializer(),
                                          regularizer=layers.l2_regularizer(self.wd))

            conv_result = tf.nn.conv2d(inputMap, conv_weight, strides=[1, stride, stride, 1], padding=padding)

            tf.summary.histogram('conv_weight', conv_weight)
            tf.summary.histogram('conv_result', conv_result)

            return conv_result

    def bnLayer(self, inputMap, scope_name, is_training):
        with tf.variable_scope(scope_name):
            return tf.layers.batch_normalization(inputMap, training=is_training, epsilon=1e-5, momentum=0.9)

    def reluLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.nn.relu(inputMap)

    def avgPoolLayer(self, inputMap, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            return tf.nn.avg_pool(inputMap, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

    def globalPoolLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            size = inputMap.get_shape()[1]
            return self.avgPoolLayer(inputMap, size, size, padding='VALID', scope_name=scope_name)

    def fcLayer(self, inputMap, out_channel, scope_name):
        with tf.variable_scope(scope_name):
            in_channel = inputMap.get_shape()[-1]
            fc_weight = tf.get_variable('fc_weight', [in_channel, out_channel],
                                        initializer=layers.variance_scaling_initializer(),
                                        regularizer=layers.l2_regularizer(self.wd))
            fc_bias = tf.get_variable('fc_bias', [out_channel], initializer=tf.zeros_initializer())

            fc_result = tf.matmul(inputMap, fc_weight) + fc_bias

            tf.summary.histogram('fc_weight', fc_weight)
            tf.summary.histogram('fc_bias', fc_bias)
            tf.summary.histogram('fc_result', fc_result)

            return fc_result

    def flattenLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.layers.flatten(inputMap)

    def residualUnitLayer(self, inputMap, out_channel, ksize, unit_name, down_sampling, is_training, first_conv=False):
        with tf.variable_scope(unit_name):
            in_channel = inputMap.get_shape().as_list()[-1]
            if down_sampling:
                stride = 2
                increase_dim = True
            else:
                stride = 1
                increase_dim = False

            if first_conv:
                conv_layer1 = self.convLayer(inputMap, out_channel, ksize, stride, scope_name='conv_layer1')
            else:
                bn_layer1 = self.bnLayer(inputMap, scope_name='bn_layer1', is_training=is_training)
                relu_layer1 = self.reluLayer(bn_layer1, scope_name='relu_layer1')
                conv_layer1 = self.convLayer(relu_layer1, out_channel, ksize, stride, scope_name='conv_layer1')

            bn_layer2 = self.bnLayer(conv_layer1, scope_name='bn_layer2', is_training=is_training)
            relu_layer2 = self.reluLayer(bn_layer2, scope_name='relu_layer2')
            conv_layer2 = self.convLayer(relu_layer2, out_channel, ksize, stride=1, scope_name='conv_layer2')

            if increase_dim:
                identical_mapping = self.avgPoolLayer(inputMap, ksize=2, stride=2, scope_name='identical_pool')
                identical_mapping = tf.pad(identical_mapping, [[0, 0], [0, 0], [0, 0],
                                                               [(out_channel - in_channel) // 2,
                                                                (out_channel - in_channel) // 2]])
            else:
                identical_mapping = inputMap

            return tf.add(conv_layer2, identical_mapping)

    def residualSectionLayer(self, inputMap, ksize, out_channel, unit_num, section_name, down_sampling, first_conv,
                             is_training):
        with tf.variable_scope(section_name):
            _out = inputMap
            _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_1', down_sampling=down_sampling,
                                          first_conv=first_conv, is_training=is_training)
            for n in range(2, unit_num + 1):
                _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_' + str(n),
                                              down_sampling=False, first_conv=False, is_training=is_training)

            return _out

    def resnet_model(self, input_x, model_name, unit_num1, unit_num2, unit_num3):
        with tf.variable_scope(model_name, reuse=tf.AUTO_REUSE):
            _conv = self.convLayer(input_x, self.oc1, self.k, stride=1, scope_name='unit1_conv')
            _bn = self.bnLayer(_conv, scope_name='unit1_bn', is_training=self.is_training)
            _relu = self.reluLayer(_bn, scope_name='unit1_relu')

            sec1_out = self.residualSectionLayer(inputMap=_relu,
                                                 ksize=self.k,
                                                 out_channel=self.oc1,
                                                 unit_num=unit_num1,
                                                 section_name='section1',
                                                 down_sampling=False,
                                                 first_conv=True,
                                                 is_training=self.is_training)

            sec2_out = self.residualSectionLayer(inputMap=sec1_out,
                                                 ksize=self.k,
                                                 out_channel=self.oc2,
                                                 unit_num=unit_num2,
                                                 section_name='section2',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            sec3_out = self.residualSectionLayer(inputMap=sec2_out,
                                                 ksize=self.k,
                                                 out_channel=self.oc3,
                                                 unit_num=unit_num3,
                                                 section_name='section3',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            _fm_bn = self.bnLayer(sec3_out, scope_name='_fm_bn', is_training=self.is_training)
            _fm_relu = self.reluLayer(_fm_bn, scope_name='_fm_relu')
            _fm_pool = self.globalPoolLayer(_fm_relu, scope_name='_fm_gap')
            _fm_flatten = self.flattenLayer(_fm_pool, scope_name='_fm_flatten')

            y_pred = self.fcLayer(_fm_flatten, self.num_class, scope_name='fc_pred')
            y_pred_softmax = tf.nn.softmax(y_pred)

            return y_pred, y_pred_softmax

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.img_h, self.img_w, 1])
        self.y = tf.placeholder(tf.int32, [None, self.num_class])
        self.is_training = tf.placeholder(tf.bool)
        tf.summary.image('input_x', self.x, max_outputs=5)

        self.y_pred, self.y_pred_softmax = self.resnet_model(input_x=self.x, model_name='ResNet', unit_num1=3,
                                                             unit_num2=3, unit_num3=3)

        with tf.variable_scope('loss'):
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.y))
            self.l2_cost = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = self.cost + self.l2_cost
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('optimize'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('tfSummary'):
            self.merged = tf.summary.merge_all()
            if self.train_phase == 'Train':
                self.writer = tf.summary.FileWriter(self.ckptDir, self.sess.graph)

        with tf.variable_scope('saver'):
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.eps)

        with tf.variable_scope('accuracy'):
            self.distribution = [tf.argmax(self.y, 1), tf.argmax(self.y_pred_softmax, 1)]
            self.correct_prediction = tf.equal(self.distribution[0], self.distribution[1])
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 'float'))

    def f_value(self, matrix):
        f = 0.0
        length = len(matrix[0])
        for i in range(length):
            recall = matrix[i][i] / np.sum([matrix[i][m] for m in range(self.num_class)])
            precision = matrix[i][i] / np.sum([matrix[n][i] for n in range(self.num_class)])
            result = (recall * precision) / (recall + precision)
            f += result
        f *= (2 / self.num_class)

        return f

    def validation_procedure(self):
        confusion_matrics = np.zeros([self.num_class, self.num_class], dtype="int")
        val_loss = 0.0

        val_batch_num = int(np.ceil(self.validation_data[0].shape[0] / self.bs))
        for step in range(val_batch_num):
            _validationImg = self.validation_data[0][step * self.bs:step * self.bs + self.bs]
            _validationLab = self.validation_data[1][step * self.bs:step * self.bs + self.bs]

            [matrix_row, matrix_col], tmp_loss = self.sess.run([self.distribution, self.loss],
                                                               feed_dict={self.x: _validationImg,
                                                                          self.y: _validationLab,
                                                                          self.is_training: False})
            for m, n in zip(matrix_row, matrix_col):
                confusion_matrics[m][n] += 1

            val_loss += tmp_loss

        validation_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(self.num_class)])) / float(
            np.sum(confusion_matrics))
        validation_loss = val_loss / val_batch_num

        return validation_accuracy, validation_loss

    def test_procedure(self):
        confusion_matrics = np.zeros([self.num_class, self.num_class], dtype="int")

        tst_batch_num = int(np.ceil(self.test_data[0].shape[0] / self.bs))
        for step in range(tst_batch_num):
            _testImg = self.test_data[0][step * self.bs:step * self.bs + self.bs]
            _testLab = self.test_data[1][step * self.bs:step * self.bs + self.bs]

            matrix_row, matrix_col = self.sess.run(self.distribution,
                                                   feed_dict={self.x: _testImg,
                                                              self.y: _testLab,
                                                              self.is_training: False})
            for m, n in zip(matrix_row, matrix_col):
                confusion_matrics[m][n] += 1

        test_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(self.num_class)])) / float(
            np.sum(confusion_matrics))
        detail_test_accuracy = [confusion_matrics[i][i] / np.sum(confusion_matrics[i]) for i in range(self.num_class)]
        log1 = "Test Accuracy : %g" % test_accuracy
        log2 = np.array(confusion_matrics.tolist())
        log3 = ''
        for j in range(self.num_class):
            log3 += 'category %s test accuracy : %g\n' % (utils.pulmonary_category[j], detail_test_accuracy[j])
        log3 = log3[:-1]
        log4 = 'F_Value : %g' % self.f_value(confusion_matrics)

        utils.save2file(log1, self.ckptDir, self.model)
        utils.save2file(log2, self.ckptDir, self.model)
        utils.save2file(log3, self.ckptDir, self.model)
        utils.save2file(log4, self.ckptDir, self.model)

    def train(self):
        print('Start to run in mode [Supervied Learning in Source Domain]')
        self.sess.run(tf.global_variables_initializer())
        self.train_itr = len(self.training_data[0]) // self.bs

        self.best_val_accuracy = []
        self.best_val_loss = []

        for e in range(1, self.eps + 1):
            _tr_img, _tr_lab = init.shuffle_data(self.training_data[0], self.training_data[1])

            training_acc = 0.0
            training_loss = 0.0

            for itr in range(self.train_itr):
                _tr_img_batch, _tr_lab_batch = init.next_batch(_tr_img, _tr_lab, self.bs, itr)
                _train_accuracy, _train_loss, _ = self.sess.run([self.accuracy, self.loss, self.train_op],
                                                                feed_dict={self.x: _tr_img_batch,
                                                                           self.y: _tr_lab_batch,
                                                                           self.is_training: True})
                training_acc += _train_accuracy
                training_loss += _train_loss

            summary = self.sess.run(self.merged, feed_dict={self.x: _tr_img_batch,
                                                            self.y: _tr_lab_batch,
                                                            self.is_training: False})

            training_acc = float(training_acc / self.train_itr)
            training_loss = float(training_loss / self.train_itr)

            validation_acc, validation_loss = self.validation_procedure()
            self.best_val_accuracy.append(validation_acc)
            self.best_val_loss.append(validation_loss)

            log1 = "Epoch: [%d], Training Accuracy: [%g], Validation Accuracy: [%g], Loss Training: [%g] " \
                   "Loss_validation: [%g], Time: [%s]" % \
                   (e, training_acc, validation_acc, training_loss, validation_loss, time.ctime(time.time()))

            self.plt_epoch.append(e)
            self.plt_training_accuracy.append(training_acc)
            self.plt_training_loss.append(training_loss)
            self.plt_validation_accuracy.append(validation_acc)
            self.plt_validation_loss.append(validation_loss)

            utils.plotAccuracy(x=self.plt_epoch,
                               y1=self.plt_training_accuracy,
                               y2=self.plt_validation_accuracy,
                               figName=self.model,
                               line1Name='training',
                               line2Name='validation',
                               savePath=self.ckptDir)

            utils.plotLoss(x=self.plt_epoch,
                           y1=self.plt_training_loss,
                           y2=self.plt_validation_loss,
                           figName=self.model,
                           line1Name='training',
                           line2Name='validation',
                           savePath=self.ckptDir)

            utils.save2file(log1, self.ckptDir, self.model)

            self.writer.add_summary(summary, e)

            self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(e))

            self.test_procedure()

        self.best_val_index = self.best_val_accuracy.index(max(self.best_val_accuracy))
        log2 = 'Highest Validation Accuracy : [%g], Epoch : [%g]' % (
            self.best_val_accuracy[self.best_val_index], self.best_val_index + 1)
        utils.save2file(log2, self.ckptDir, self.model)

        self.best_val_index_loss = self.best_val_loss.index(min(self.best_val_loss))
        log3 = 'Lowest Validation Loss : [%g], Epoch : [%g]' % (
            self.best_val_loss[self.best_val_index_loss], self.best_val_index_loss + 1)
        utils.save2file(log3, self.ckptDir, self.model)

    def test(self):
        print('Start to run in mode [Test in Target Domain]')
        self.saver.restore(self.sess, self.ckptDir + self.model + '-' + str(self.res_eps))
        self.test_procedure()
