def attention_block(input, block_name):
    with tf.variable_scope(block_name):
        input_shape = input.get_shape().as_list()

        mask_downsample1 = avg_pool_layer(input, ksize=2, stride=2, layer_name='mask_downsample1')
        mask_residual1 = residual_unit(input=mask_downsample1,
                                       out_channel=input_shape[-1],
                                       ksize=3,
                                       unit_name='mask_residual1',
                                       down_sampling=False)

        skip = residual_unit(input=mask_residual1,
                             out_channel=input_shape[-1],
                             ksize=3,
                             unit_name='skip',
                             down_sampling=False)

        mask_downsample2 = avg_pool_layer(mask_residual1, ksize=2, stride=2, layer_name='mask_downsample2')
        mask_residual2 = residual_unit(input=mask_downsample2,
                                       out_channel=input_shape[-1],
                                       ksize=3,
                                       unit_name='mask_residual2',
                                       down_sampling=False)

        mask_upsample1 = tf.image.resize_bilinear(mask_residual2, size=[input_shape[1] // 2, input_shape[2] // 2])

        fusion = tf.add(skip, mask_upsample1)

        mask_residual4 = residual_unit(input=fusion,
                                       out_channel=input_shape[-1],
                                       ksize=3,
                                       unit_name='mask_residual4',
                                       down_sampling=False)
        mask_upsample2 = tf.image.resize_bilinear(mask_residual4, size=[input_shape[1], input_shape[2]])

        bn1 = bn_layer(mask_upsample2, layer_name='bn1')
        relu1 = relu_layer(bn1, layer_name='relu1')
        conv1 = conv_layer(relu1, out_channel=input_shape[-1], ksize=1, stride=1, layer_name='conv1')

        bn2 = bn_layer(conv1, layer_name='bn2')
        relu2 = relu_layer(bn2, layer_name='relu2')
        conv2 = conv_layer(relu2, out_channel=input_shape[-1], ksize=1, stride=1, layer_name='conv2')

        sigmoid = tf.nn.sigmoid(conv2)
        image_summary(sigmoid, layer_name='attention_img')

        attention_result = tf.multiply(input, sigmoid)

        image_summary(attention_result, layer_name='attention_mask')

        attention_residual = tf.add(attention_result, input)

    return attention_residual


def spp_block(input, block_name):
    with tf.variable_scope(block_name):
        img_shape = input.get_shape().as_list()

        pool4x4_kernel = np.ceil(img_shape[1] / 4)
        pool4x4_stride = np.floor(img_shape[1] / 4)

        pool2x2_kernel = np.ceil(img_shape[1] / 2)
        pool2x2_stride = np.floor(img_shape[1] / 2)

        pool1x1_kernel = img_shape[1]
        pool1x1_stride = img_shape[1]

        maxpool_4x4 = max_pool_layer(input, ksize=pool4x4_kernel, stride=pool4x4_stride, layer_name='spp_pool4x4')
        maxpool_2x2 = max_pool_layer(input, ksize=pool2x2_kernel, stride=pool2x2_stride, layer_name='spp_pool2x2')
        maxpool_1x1 = max_pool_layer(input, ksize=pool1x1_kernel, stride=pool1x1_stride, layer_name='spp_pool1x1')

        flatten_4x4 = tf.layers.flatten(maxpool_4x4)
        flatten_2x2 = tf.layers.flatten(maxpool_2x2)
        flatten_1x1 = tf.layers.flatten(maxpool_1x1)

        concated = tf.concat([flatten_4x4, flatten_2x2, flatten_1x1], axis=-1)

    return concated


def attention_block(input, block_name):
    with tf.variable_scope(block_name):
        in_channel = input.get_shape().as_list()[-1]
        bn1 = bn_layer(input, layer_name='bn1')
        relu1 = relu_layer(bn1, layer_name='relu1')
        global_pool = global_avg_pool_layer(input=relu1, layer_name='global_pool')
        global_pool = flatten(global_pool, layer_name='flatten')
        fc1 = fc_layer(input=global_pool, out_channel=in_channel // 2, layer_name='fc1')
        bn2 = bn_layer(fc1, layer_name='bn2')
        relu2 = relu_layer(bn2, layer_name='relu2')
        fc2 = fc_layer(relu2, out_channel=in_channel, layer_name='fc2')
        sigmoid = tf.nn.sigmoid(fc2)
        tf.summary.histogram('attention_sigmoid', sigmoid)

        sig_reshape = tf.reshape(sigmoid, [-1, 1, 1, in_channel])
        attention_mask = input * sig_reshape
        image_summary(attention_mask, layer_name='attention_mask')
        attention_residual = input + attention_mask

    return attention_residual