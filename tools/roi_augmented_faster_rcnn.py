#!/usr/bin/env python

# --------------------------------------------------
# Gyrfalcontech Faster RCNN
# 
# Augments ROIs for a 224x224 detection by sliding
# 224x224 window across higher resolution image and
# concatenating the low res and high res ROIs.
# 
# Author: Jonathan Zhang, 2017
# --------------------------------------------------


import _init_paths
from fast_rcnn.test import im_list_to_blob, vis_detections
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from fast_rcnn.nms_wrapper import nms
from datasets.factory import get_imdb
from utils.timer import Timer
from caffe_utils import ceil, setup_net_inputs, process_net_outputs
from caffe_utils import preprocess_img
import caffe
import argparse
import pprint
import numpy as np
import cv2
import os
import sys


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
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--vis', dest='vis', help='enable visual debugging',
                        action="store_true")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def vis_debug_boxes(im, scores, boxes, imdb, thresh=0.05):
    for j in xrange(1, imdb.num_classes):

        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j * 4:(j + 1) * 4]

        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)

        keep = nms(cls_dets, cfg.TEST.NMS)
        cls_dets = cls_dets[keep, :]
        vis_detections(im, imdb.classes[j], cls_dets)


def get_rois_pyramid(net, im, window_size=224, min_ov=100, scale=(672, 1000),
                     vis=False):
    '''Get rois from 224x224 sliding window slices of the higher resolution
    image.

    net: caffe.Net instance
    im: original input image
    '''
    cfg.TEST.RPN_POST_NMS_TOP_N = 50
    cfg.TEST.RPN_MIN_SIZE = 16
    cfg.BBOX_REG = True

    net_img, im_scale = preprocess_img(im, scale[0], scale[1])
    im_height, im_width = net_img.shape[:2]

    ov_h = ov_w = min_ov
    # ov_h, ov_w = ajudt_overlap(im_height, im_width, window_size, min_ov)
    # Need to add 1 to account for the subtraction earlier
    rows = ceil((im_height - window_size) / ov_h) + 1
    cols = ceil((im_width - window_size) / ov_w) + 1

    outlen = cfg.TEST.RPN_POST_NMS_TOP_N
    total = rows * cols * outlen
    all_scores = np.zeros((total, 21))
    all_boxes = np.zeros((total, 84))
    all_rois = np.zeros((total, 5))

    for i in xrange(rows):
        # calculate range of rows to slice (max: window_size)
        row_start = min(i * ov_h, im_height - window_size)
        row_end = row_start + window_size
        for j in xrange(cols):
            # calculate range of columns to slice (max: window_size)
            col_start = min(j * ov_w, im_width - window_size)
            col_end = col_start + window_size

            # slice out the appropriate portion and make copy
            img_slice = net_img[row_start:row_end, col_start:col_end].astype(
                np.float32, copy=True)
            # plt.imshow(img_slice[:, :, (2, 1, 0)] / 255)
            # plt.show()
            img_blob = im_list_to_blob([img_slice])

            # set im_scale to 1 (i.e.) do not rescale to original image space
            kwargs = setup_net_inputs(net, img_blob, im_scale)
            blobs_out = net.forward(**kwargs)
            scores, boxes, rois = process_net_outputs(net, blobs_out,
                                                      img_slice.shape,
                                                      im_scale)
            # rois are different from boxes in that the boxes have been through
            # bounding box regressions and mapped back to original image space

            # [width, height]
            # add offsets to map back to entire image
            # boxes are NOT mapped back to the original image space
            dx, dy = col_start / im_scale, row_start / im_scale
            boxes += np.tile([dx, dy], 42)

            # map rois back to original image space
            rois += np.asarray([0, dx, dy, dx, dy])

            # concatenate all boxes/rois/scores into respective arrays
            inds = (slice((i * j + j) * outlen, (i * j + j + 1) * outlen),
                    slice(None))
            all_scores[inds] = scores[...]
            all_boxes[inds] = boxes[...]
            all_rois[inds] = rois[...]

    return all_scores, all_boxes, all_rois


def roi_augmented_detection(net, im, hr_rois, scale=(672, 1000), vis=False):
    '''Adds rois generated from higher resolution image to the rois from
    the low resolution 224x224 image and returns scores and bounding boxes.
    

    net: caffe.Net instance
    im: original image
    hr_rois: high resolution rois
    '''
    cfg.TEST.RPN_POST_NMS_TOP_N = 300
    net_img, im_scale = preprocess_img(im, scale[0], scale[1])
    img_blob = im_list_to_blob([net_img])

    kwargs = setup_net_inputs(net, img_blob, im_scale)
    # run forward pass until proposal layer
    rois_blob = net.forward(end="proposal", **kwargs)
    lr_rois = rois_blob['rois']
    # map higher resolution rois onto lower resolution space - there may be lots
    # of redundancy
    hr_rois *= im_scale
    all_rois = np.vstack((lr_rois, hr_rois))

    net.blobs['rois'].reshape(*(all_rois.shape))
    net.blobs['rois'].data[...] = all_rois.astype(np.float32, copy=False)

    blobs_out = net.forward(start="roi_pool5")
    scores, boxes, rois = process_net_outputs(net, blobs_out, net_img.shape,
                                              im_scale)

    return scores, boxes


def image_pyramid_detection(net, im, scales, imdb, args, _t):
    # "Stage 1: Getting ROIS from high resolution pyramid"
    _t['stage1'].tic()
    hr_scores, hr_boxes, hr_rois = get_rois_pyramid(net, im, vis=args.vis)
    _t['stage1'].toc()

    # print "stage 2: Compiling ROIS for low resolution RCNN"
    _t['stage2'].tic()
    scores, boxes = roi_augmented_detection(net, im, hr_rois, vis=args.vis)
    _t['stage2'].toc()
    return scores, boxes
    # return scores, pred_boxes


def test_net_img_pyr(net, imdb, args, max_per_image=100, thresh=0.05):
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    _t = {"stage1": Timer(), "stage2": Timer()}

    for k in xrange(num_images):

        im = cv2.imread(imdb.image_path_at(k))
        scales = [(224, 224), (672, 1000)]
        scores, boxes = image_pyramid_detection(net, im, scales, imdb, args, _t)

        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if args.vis:
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

        print "\n{}/{}\t{:.3f}s\t{:.3f}s".format(k + 1, num_images,
                                                 _t["stage1"].average_time,
                                                 _t["stage2"].average_time)

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

    test_net_img_pyr(net, imdb, args)
