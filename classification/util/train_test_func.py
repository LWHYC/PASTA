# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
from data_process.data_process_func import *
import sys

def process_bar(num, total):
    rate = float(num) / total
    progress = int(50 * rate)
    bar = '█' * progress + '-' * (50 - progress)
    percent = format(rate * 100, '.2f')
    r = '\r[{}]{:>5}% ({}/{})'.format(bar, percent, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def volume_probability_prediction(temp_imgs, data_shape, label_shape, data_channel,
                                  class_num, batch_size, sess, proby, x):
    '''
    Test one image with sub regions along z-axis
    '''
    [D, H, W] = temp_imgs[0].shape
    input_center = [int(D/2), int(H/2), int(W/2)]
    temp_prob = np.zeros([D, H, W, class_num])
    sub_image_baches = []
    for center_slice in range(int(label_shape[0]/2), D + int(label_shape[0]/2), label_shape[0]):
        center_slice = min(center_slice, D - int(label_shape[0]/2))
        sub_image_bach = []
        for chn in range(data_channel):
            temp_input_center = [center_slice, input_center[1], input_center[2]]
            sub_image = extract_roi_from_volume(
                            temp_imgs[chn], temp_input_center, data_shape)
            sub_image_bach.append(sub_image)
        sub_image_bach = np.asanyarray(sub_image_bach, np.float32)
        sub_image_baches.append(sub_image_bach)
    total_batch = len(sub_image_baches)
    max_mini_batch = int((total_batch+batch_size-1)/batch_size)
    sub_label_idx = 0
    for mini_batch_idx in range(max_mini_batch):
        data_mini_batch = sub_image_baches[mini_batch_idx*batch_size:
                                      min((mini_batch_idx+1)*batch_size, total_batch)]
        if(mini_batch_idx == max_mini_batch - 1):
            for idx in range(batch_size - (total_batch - mini_batch_idx*batch_size)):
                data_mini_batch.append(np.random.normal(0, 1, size = [data_channel] + data_shape))
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
        prob_mini_batch = sess.run(proby, feed_dict = {x:data_mini_batch})
        
        for batch_idx in range(prob_mini_batch.shape[0]):
            center_slice = sub_label_idx*label_shape[0] + int(label_shape[0]/2)
            center_slice = min(center_slice, D - int(label_shape[0]/2))
            temp_input_center = [center_slice, input_center[1], input_center[2], int(class_num/2)]
            sub_prob = np.reshape(prob_mini_batch[batch_idx], label_shape + [class_num])
            temp_prob = set_roi_to_volume(temp_prob, temp_input_center, sub_prob)
            sub_label_idx = sub_label_idx + 1
    return temp_prob 


def volume_probability_prediction_3d_roi(temp_imgs, data_shape, label_shape, data_channel, 
                                  class_num, batch_size, sess, proby, x):
    '''
    Test one image with sub regions along x, y, z axis
    '''
    [D, H, W] = temp_imgs[0].shape
    temp_prob = np.zeros([D, H, W, class_num])
    sub_image_batches = []
    sub_image_centers = []
    roid_half = int(label_shape[0]/2)
    roih_half = int(label_shape[1]/2)
    roiw_half = int(label_shape[2]/2)
    for centerd in range(roid_half, D + roid_half, label_shape[0]):
        centerd = min(centerd, D - roid_half)
        for centerh in range(roih_half, H + roih_half, label_shape[1]):
            centerh =  min(centerh, H - roih_half) 
            for centerw in range(roiw_half, W + roiw_half, label_shape[2]):
                centerw =  min(centerw, W - roiw_half) 
                temp_input_center = [centerd, centerh, centerw]
                sub_image_centers.append(temp_input_center)
                sub_image_batch = []
                for chn in range(data_channel):
                    sub_image = extract_roi_from_volume(temp_imgs[chn], temp_input_center, data_shape)
                    sub_image_batch.append(sub_image)
                sub_image_bach = np.asanyarray(sub_image_batch, np.float32)
                sub_image_batches.append(sub_image_bach)

    total_batch = len(sub_image_batches)
    max_mini_batch = int((total_batch + batch_size - 1)/batch_size)
    sub_label_idx = 0
    for mini_batch_idx in range(max_mini_batch):
        data_mini_batch = sub_image_batches[mini_batch_idx*batch_size:
                                      min((mini_batch_idx+1)*batch_size, total_batch)]
        if(mini_batch_idx == max_mini_batch - 1):
            for idx in range(batch_size - (total_batch - mini_batch_idx*batch_size)):
                data_mini_batch.append(np.random.normal(0, 1, size = [data_channel] + data_shape))
        data_mini_batch = np.asanyarray(data_mini_batch, np.float32)
        data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
        outprob_mini_batch = sess.run(proby, feed_dict = {x:data_mini_batch})
        
        for batch_idx in range(batch_size):
            glb_batch_idx = batch_idx + mini_batch_idx * batch_size
            if(glb_batch_idx >= total_batch):
                continue
            temp_center = sub_image_centers[glb_batch_idx]
            temp_prob = set_roi_to_volume(temp_prob, temp_center + [1], outprob_mini_batch[batch_idx])
            sub_label_idx = sub_label_idx + 1
    return temp_prob 
   
