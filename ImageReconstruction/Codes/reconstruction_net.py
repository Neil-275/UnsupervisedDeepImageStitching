import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.activations import relu, tanh

def resBlock(x):
    conv1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    conv2 = Conv2D(filters=64, kernel_size=3, padding='same', activation=None)(conv1)
    out = relu(x + conv2)
    return out

def ReconstructionNet(inputs):
    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    inputs.set_shape([1, None, None, 6])
    
    HR_inputs = inputs

    ########################################################
    ################### low-resolution branch###############
    ########################################################
    # the input of low-resolution branch
    warp1 = tf.image.resize(inputs[..., 0:3], [256, 256], method='bilinear')
    warp2 = tf.image.resize(inputs[..., 3:6], [256, 256], method='bilinear')
    LR_inputs = tf.concat([warp1, warp2], axis=3)

    # low-resolution reconstruction branch (encoder)
    encoder_conv1_1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(LR_inputs)
    encoder_conv1_2 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(encoder_conv1_1)
    encoder_pooling1 = MaxPooling2D(pool_size=2, padding='same')(encoder_conv1_2)

    encoder_conv2_1 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(encoder_pooling1)
    encoder_conv2_2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(encoder_conv2_1)
    encoder_pooling2 = MaxPooling2D(pool_size=2, padding='same')(encoder_conv2_2)

    encoder_conv3_1 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(encoder_pooling2)
    encoder_conv3_2 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(encoder_conv3_1)
    encoder_pooling3 = MaxPooling2D(pool_size=2, padding='same')(encoder_conv3_2)

    encoder_conv4_1 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(encoder_pooling3)
    encoder_conv4_2 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(encoder_conv4_1)

    # low-resolution reconstruction branch (decoder)
    decoder_up1 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding='same')(encoder_conv4_2)
    decoder_concat1 = tf.concat([encoder_conv3_2, decoder_up1], axis=3)
    decoder_conv1_1 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(decoder_concat1)
    decoder_conv1_2 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(decoder_conv1_1)

    decoder_up2 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding='same')(decoder_conv1_2)
    decoder_concat2 = tf.concat([encoder_conv2_2, decoder_up2], axis=3)
    decoder_conv2_1 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(decoder_concat2)
    decoder_conv2_2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(decoder_conv2_1)

    decoder_up3 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding='same')(decoder_conv2_2)
    decoder_concat3 = tf.concat([encoder_conv1_2, decoder_up3], axis=3)
    decoder_conv3_1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(decoder_concat3)
    decoder_conv3_2 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(decoder_conv3_1)

    # the output of low-resolution branch
    LR_output = Conv2D(filters=3, kernel_size=3, padding='same', activation=None)(decoder_conv3_2)
    LR_output = tanh(LR_output)

    ########################################################
    ################### high-resolution branch###############
    ########################################################
    # the input of high-resolution branch
    LR_SR = tf.image.resize(LR_output, [height, width], method='bilinear')
    HR_inputs = tf.concat([HR_inputs, LR_SR], axis=3)

    # high-resolution reconstruction branch
    HR_conv1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(HR_inputs)
    x = HR_conv1
    for i in range(8):
        x = resBlock(x)
    HR_conv2 = Conv2D(filters=64, kernel_size=3, padding='same', activation=None)(x)
    HR_conv2 = relu(HR_conv1 + HR_conv2)

    # the output of high-resolution branch
    HR_output = Conv2D(filters=3, kernel_size=3, padding='same', activation=None)(HR_conv2)
    HR_output = tanh(HR_output)

    return LR_output, HR_output