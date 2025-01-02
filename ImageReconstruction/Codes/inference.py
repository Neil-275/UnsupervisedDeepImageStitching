import tensorflow as tf
import numpy as np
import cv2
import os

from models import reconstruction
from utils import DataLoader, load, save
import constant

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER
snapshot_dir = constant.SNAPSHOT_DIR + '/model.ckpt-200000'
batch_size = constant.TEST_BATCH_SIZE

# Define dataset
data_loader = DataLoader(test_folder)

# Build the model
test_inputs = tf.keras.Input(shape=(None, None, 3 * 2), dtype=tf.float32)
lr_test_stitched, hr_test_stitched = reconstruction(test_inputs)

# Create a model
model = tf.keras.Model(inputs=test_inputs, outputs=[lr_test_stitched, hr_test_stitched])

# Load model weights
model.load_weights(snapshot_dir)

def inference_func():
    print("Performing inference...")
    length = 1106
    
    for i in range(length):
        input_clip = np.expand_dims(data_loader.get_image_clips(i), axis=0)
        lr_result, stitch_result = model.predict(input_clip)
        
        stitch_result = (stitch_result + 1) * 127.5    
        stitch_result = stitch_result[0]
        path = "../results/" + str(i + 1).zfill(6) + ".jpg"
        cv2.imwrite(path, stitch_result)
        print('i = {} / {}'.format(i + 1, length))
        
    print("===================DONE!==================")  

inference_func()