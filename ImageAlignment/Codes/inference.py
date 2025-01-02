import tensorflow as tf
import os
import numpy as np
import cv2 as cv
import skimage
from models import H_estimator
from utils import DataLoader, load, save
import constant

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER
snapshot_dir = constant.SNAPSHOT_DIR + '/model.ckpt-1000000'
batch_size = constant.TEST_BATCH_SIZE

# Define dataset
data_loader = DataLoader(test_folder)

# Build the model
test_inputs = tf.keras.Input(shape=(128, 128, 3 * 2), dtype=tf.float32)
test_net1_f, test_net2_f, test_net3_f, test_warp2_H1, test_warp2_H2, test_warp2_H3, test_one_warp_H1, test_one_warp_H2, test_one_warp_H3 = H_estimator(test_inputs, test_inputs, False)

# Load model weights
model = tf.keras.Model(inputs=test_inputs, outputs=[test_net1_f, test_net2_f, test_net3_f, test_warp2_H1, test_warp2_H2, test_warp2_H3, test_one_warp_H1, test_one_warp_H2, test_one_warp_H3])
model.load_weights(snapshot_dir)

def inference_func():
    print("Performing inference...")
    length = 1106
    psnr_list = []
    ssim_list = []

    for i in range(length):
        # Load test data
        input_clip = np.expand_dims(data_loader.get_data_clips(i, 128, 128), axis=0)

        # Inference
        outputs = model(input_clip)
        warp = (outputs[5] + 1) * 127.5
        warp_one = outputs[7][0]

        input1 = (input_clip[..., 0:3] + 1) * 127.5[0]
        input2 = (input_clip[..., 3:6] + 1) * 127.5[0]

        # Compute PSNR/SSIM
        psnr = skimage.metrics.peak_signal_noise_ratio(input1 * warp_one, warp * warp_one, data_range=255)
        ssim = skimage.metrics.structural_similarity(input1 * warp_one, warp * warp_one, data_range=255, multichannel=True)

        print('i = {} / {}, psnr = {:.6f}'.format(i + 1, length, psnr))

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print("=================== Results Analysis ===================")
    
    psnr_list.sort(reverse=True)
    psnr_list_30 = psnr_list[0:331]
    psnr_list_60 = psnr_list[331:663]
    psnr_list_100 = psnr_list[663:]
    print("top 30%:", np.mean(psnr_list_30))
    print("top 30~60%:", np.mean(psnr_list_60))
    print("top 60~100%:", np.mean(psnr_list_100))
    print('average psnr:', np.mean(psnr_list))

    ssim_list.sort(reverse=True)
    ssim_list_30 = ssim_list[0:331]
    ssim_list_60 = ssim_list[331:663]
    ssim_list_100 = ssim_list[663:]
    print("top 30%:", np.mean(ssim_list_30))
    print("top 30~60%:", np.mean(ssim_list_60))
    print("top 60~100%:", np.mean(ssim_list_100))
    print('average ssim:', np.mean(ssim_list))

inference_func()