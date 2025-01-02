import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import time

def Vgg19_simple(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image tensor [batch, height, width, 3] values scaled [0, 1]
    reuse : boolean, whether to reuse the variables
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    print("Reuse:", reuse)

    start_time = time.time()
    print("Build model started")

    rgb_scaled = rgb * 255.0

    # Convert RGB to BGR
    red, green, blue = tf.split(rgb_scaled, 3, axis=-1)

    # Normalize the BGR channels
    bgr = tf.concat(
        [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], axis=-1
    )

    """ Input layer """
    inputs = Input(shape=(None, None, 3), name='input')
    
    """ Convolutional Layers """
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_1')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_3')(x)
    conv_low = x
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_3')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_3')(x)
    conv_high = x
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool5')(x)

    """ Fully Connected Layers """
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dense(1000, activation='linear', name='fc8')(x)

    print("Build model finished: %fs" % (time.time() - start_time))
    return conv_high, conv_low