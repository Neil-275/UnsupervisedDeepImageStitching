import tensorflow as tf
import os
import numpy as np
import cv2
from models import H_estimator, output_H_estimator
from utils import DataLoader, load, save
import constant
import skimage

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
train_folder = constant.TRAIN_FOLDER
test_folder = constant.TEST_FOLDER
snapshot_dir = constant.SNAPSHOT_DIR + '/model.ckpt-1000000'
batch_size = constant.TEST_BATCH_SIZE

# Define dataset
data_loader_train = DataLoader(train_folder)
data_loader_test = DataLoader(test_folder)

# Build the model
test_inputs = tf.keras.Input(shape=(None, None, 3 * 2), dtype=tf.float32)
test_size = tf.keras.Input(shape=(batch_size, 2, 1), dtype=tf.float32)
test_coarsealignment = output_H_estimator(test_inputs, test_size, False)

model = tf.keras.Model(inputs=[test_inputs, test_size], outputs=test_coarsealignment)

# Load model weights
model.load_weights(snapshot_dir)

def inference_func():
    print("Performing inference...")

    # Generating aligned images for training set
    print("------------------------------------------")
    print("Generating aligned images for training set")
    length = 10440
    for i in range(length):
        input_clip = np.expand_dims(data_loader_train.get_data_clips(i, None, None), axis=0)
        size_clip = np.expand_dims(data_loader_train.get_size_clips(i), axis=0)
        
        coarsealignment = model.predict([input_clip, size_clip])
        
        coarsealignment = coarsealignment[0]
        warp1 = (coarsealignment[..., 0:3] + 1.) * 127.5
        warp2 = (coarsealignment[..., 3:6] + 1.) * 127.5
        mask1 = coarsealignment[..., 6:9] * 255
        mask2 = coarsealignment[..., 9:12] * 255
        
        cv2.imwrite(f'../output/training/warp1/{str(i+1).zfill(6)}.jpg', warp1)
        cv2.imwrite(f'../output/training/warp2/{str(i+1).zfill(6)}.jpg', warp2)
        cv2.imwrite(f'../output/training/mask1/{str(i+1).zfill(6)}.jpg', mask1)
        cv2.imwrite(f'../output/training/mask2/{str(i+1).zfill(6)}.jpg', mask2)
        
        print(f'i = {i+1} / {length}')

    print("-----------Training set done--------------")
    print("------------------------------------------")
    
    # Generating aligned images for testing set
    print("------------------------------------------")
    print("Generating aligned images for testing set")
    length = 1106
    for i in range(length):
        input_clip = np.expand_dims(data_loader_test.get_data_clips(i, None, None), axis=0)
        size_clip = np.expand_dims(data_loader_test.get_size_clips(i), axis=0)
        
        coarsealignment = model.predict([input_clip, size_clip])
        
        coarsealignment = coarsealignment[0]
        warp1 = (coarsealignment[..., 0:3] + 1.) * 127.5
        warp2 = (coarsealignment[..., 3:6] + 1.) * 127.5
        mask1 = coarsealignment[..., 6:9] * 255
        mask2 = coarsealignment[..., 9:12] * 255
        
        cv2.imwrite(f'../output/testing/warp1/{str(i+1).zfill(6)}.jpg', warp1)
        cv2.imwrite(f'../output/testing/warp2/{str(i+1).zfill(6)}.jpg', warp2)
        cv2.imwrite(f'../output/testing/mask1/{str(i+1).zfill(6)}.jpg', mask1)
        cv2.imwrite(f'../output/testing/mask2/{str(i+1).zfill(6)}.jpg', mask2)
        
        print(f'i = {i+1} / {length}')

    print("-----------Testing set done--------------")
    print("------------------------------------------")

inference_func()