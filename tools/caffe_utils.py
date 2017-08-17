# ----------------------------------------------
# Gyrfalcontech Faster RCNN
# 
# Utility functions for ease of use with
# py-faster-rcnn testing.
# 
# Author: Jonathan Zhang, 2017
# ----------------------------------------------
import _init_paths
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.config import cfg
import math
import numpy as np
import cv2


def ceil(x):
    '''Integer ceilin function
    '''
    return int(math.ceil(x))


def preprocess_img(im, target_size=cfg.TEST.SCALES[0],
                   max_size=cfg.TEST.MAX_SIZE, im_means=cfg.PIXEL_MEANS):
    '''Resizes image so that shorter side is target_size and imits max image
    side to max_size. Subtracts the image mean from image.

    im: ndarray image
    target_size: int
    max_size: int
    '''
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= im_means

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = float(target_size / float(im_size_min))
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size / float(im_size_max))
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    return im, im_scale


def im_list_to_blob(ims):
    """Convert a list of images into a network input blob.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def setup_net_inputs(net, im_blob, im_scale):
    '''Prepares the network for forward pass by setting up input shapes and data
    Returns kwargs, a dictionary of keyword values to pass into net.forward

    net: caffe.Net instance
    im_blob: input image in proper blob format
    im_scale: resizing factor from original to input image (a return value from
              preprocess_image)
    '''
    blob = {'data': im_blob}
    blob["im_info"] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scale]], dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blob['data'].shape))
    net.blobs['im_info'].reshape(*(blob['im_info'].shape))

    # prepare forward arguments
    forward_kwargs = {
        'data': blob['data'].astype(np.float32, copy=False),
        'im_info': blob['im_info'].astype(np.float32, copy=False)
    }
    return forward_kwargs


def process_net_outputs(net, blobs_out, im_shape, im_scale):
    '''Retrieves network outputs after forward pass and does post-processing
    to bounding box coordinates. Returns scores, predicted boxes, and rois.

    net: caffe.Net instsances
    blobs_out: dictionary result of forward pass through network
    im_shape: shape of original image
    im_scale: resizing factor from original to input image (a return value from
              preprocess_image)
    '''
    rois = net.blobs['rois'].data.copy()

    # map back to original image dimensions
    boxes = rois[:, 1:5] / im_scale

    scores = blobs_out['cls_prob']

    # apply bounding box regression
    box_deltas = blobs_out['bbox_pred']
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_shape)

    # otherwise, just repeat boxes for all shapes
    # pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes, rois
