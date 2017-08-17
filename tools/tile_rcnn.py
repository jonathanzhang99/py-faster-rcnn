#!/usr/bin/env python

# --------------------------------------------------
# Gyrfalcontech Faster RCNN
# 
# Runs Tile RCNN. Refer to documentation/presentation
# for description of algorithm
# 
# Author: Jonathan Zhang, 2017
# --------------------------------------------------

import _init_paths
from fast_rcnn.test import _get_blobs, vis_detections
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
from datasets.factory import get_imdb
from utils.timer import Timer
from caffe_utils import ceil, setup_net_inputs, process_net_outputs
import caffe
import argparse
import pprint
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import time

import cPickle


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--vis', dest='vis', help='show detections',
                        default=False, action='store_true')
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def create_tiled_feature_map(net, im_blob, im_scale, window_size=224,
                             stride=16):
    conv_final = np.zeros((1, 512, ceil(im_blob.shape[2] / 16.0),
                           ceil(im_blob.shape[3] / 16.0)))

    im_height = im_blob.shape[2]
    im_width = im_blob.shape[3]

    # Need to find ceiling of integer division
    num_height_windows = (im_height - window_size + stride - 1) / stride + 1
    num_width_windows = (im_width - window_size + stride - 1) / stride + 1

    for j in xrange(num_height_windows):
        h_stride = min(j * stride, im_height - window_size)
        for i in xrange(num_width_windows):
            w_stride = (min(i * stride, im_width - window_size))

            window_blob = {}

            window_blob['data'] = im_blob[:, :,
                                          h_stride: h_stride + window_size,
                                          w_stride: w_stride + window_size]
            assert window_blob['data'].shape == (1, 3, 224, 224), \
                (h_stride, w_stride, im_blob.shape)

            window_blob['im_info'] = np.array([[im_height, im_width, im_scale]])
            net.blobs['data'].reshape(*(window_blob['data'].shape))
            net.blobs['im_info'].reshape(*(window_blob['im_info'].shape))

            forward_kwargs = {}
            forward_kwargs['data'] = window_blob['data'].astype(
                np.float32, copy=True)
            forward_kwargs['im_info'] = window_blob['im_info'].astype(
                np.float32, copy=False)

            blobs_out = net.forward(end='relu5_3', **forward_kwargs)
            dx = window_size / 16
            h0 = h_stride / 16
            w0 = w_stride / 16
            map_slice = conv_final[:, :, h0:h0 + dx, w0:w0 + dx]
            assert blobs_out['conv5_3'].shape[2:] == (dx, dx), \
                (blobs_out['conv5_3'].shape, window_blob['data'].shape)

            np.maximum(map_slice, blobs_out['conv5_3'], out=map_slice)


def im_detect(net, im, _t, window_size=224, stride=64):
    '''
    '''
    all_fms = []

    sz1 = 224
    im1 = im.copy()

    cfg.TEST.SCALES = (sz1,)
    cfg.TEST.MAX_SIZE = 1000
    blobs, im_scales = _get_blobs(im1, None)

    im_blob = blobs['data']

    # generate the feature map of original image for later comparison
    kwargs = setup_net_inputs(net, im_blob, im_scales[0])
    blobs_out = net.forward(**kwargs)
    b1 = net.blobs['conv5_3'].data.copy()

    conv_final = np.zeros((1, 512, ceil(im_blob.shape[2] / 16.0),
                           ceil(im_blob.shape[3] / 16.0)))

    im_height = im_blob.shape[2]
    im_width = im_blob.shape[3]

    # Need to find ceiling of integer division
    num_height_windows = (im_height - window_size + stride - 1) / stride + 1
    num_width_windows = (im_width - window_size + stride - 1) / stride + 1

    for j in xrange(num_height_windows):
        h_stride = min(j * stride, im_height - window_size)
        for i in xrange(num_width_windows):
            w_stride = (min(i * stride, im_width - window_size))

            window_blob = {}

            window_blob['data'] = im_blob[:, :,
                                          h_stride: h_stride + window_size,
                                          w_stride: w_stride + window_size]
            assert window_blob['data'].shape == (1, 3, 224, 224), \
                (h_stride, w_stride, im_blob.shape)

            window_blob['im_info'] = np.array([[im_height, im_width,
                                                im_scales[0]]])
            net.blobs['data'].reshape(*(window_blob['data'].shape))
            net.blobs['im_info'].reshape(*(window_blob['im_info'].shape))

            forward_kwargs = {}
            forward_kwargs['data'] = window_blob['data'].astype(
                np.float32, copy=True)
            forward_kwargs['im_info'] = window_blob['im_info'].astype(
                np.float32, copy=False)

            blobs_out = net.forward(end='relu5_3', **forward_kwargs)
            dx = window_size / 16
            h0 = h_stride / 16
            w0 = w_stride / 16
            map_slice = conv_final[:, :, h0:h0 + dx, w0:w0 + dx]
            assert blobs_out['conv5_3'].shape[2:] == (dx, dx), \
                (blobs_out['conv5_3'].shape, window_blob['data'].shape)

            np.maximum(map_slice, blobs_out['conv5_3'], out=map_slice)

    for i in xrange(0):
        plt.figure(num="original")
        plt.imshow(b1[0, i, ...])
        for j, t in enumerate(all_fms):
            plt.figure(num="maps: " + str(j))
            plt.imshow(t[0, i, ...])
        plt.figure(num='artificial conv')
        plt.imshow(conv_final[0, i, ...])

        plt.show()
        print "shown"
    _t["conv"].toc()

    _t["fc"].tic()
    net.blobs['conv5_3'].reshape(*(conv_final.shape))
    net.blobs['conv5_3'].data[...] = conv_final.astype(np.float32, copy=False)
    out = net.forward(start='relu5_3')
    _t["fc"].toc()

    scores, boxes = process_net_outputs(net, out, im.shape, im_scales[0])
    return scores, boxes


def test_net(net, imdb, max_per_image=100, thresh=0.05, window_size=224,
             stride=32, vis=False):
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    _t = {"conv": Timer(), "fc": Timer()}

    for k in xrange(num_images):

        im = cv2.imread(imdb.image_path_at(k))
        scores, boxes = im_detect(net, im, _t, window_size, stride)
        print scores.shape, boxes.shape

        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets)
            all_boxes[j][k] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][k][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][k][:, -1] >= image_thresh)[0]
                    all_boxes[j][k] = all_boxes[j][k][keep, :]

        print "{}/{}\t{:.3f}s\t{:.3f}s\t{:.3f}s".format(
            k + 1, num_images, _t["conv"].average_time,
            _t["overlap"].average_time, _t["fc"].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, caffe.TEST, weights=args.caffemodel)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = get_imdb(args.imdb_name)

    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    test_net(net, imdb, stride=16)
