# --------------------------------------------------
# Gyrfalcontech Faster RCNN
# 
# Uses a Region Proposer network with 224x224 input
# and crops ROIs from original image, which are
# then individually classified and go through
# bounding box regression
# 
# Author: Jonathan Zhang, 2017
# --------------------------------------------------

import _init_paths
import caffe
import numpy as np
import argparse
import sys
import cv2
import pprint

from caffe_utils import preprocess_img
from caffe_utils import setup_net_inputs, process_net_outputs
from fast_rcnn.test import im_list_to_blob, vis_detections
from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.nms_wrapper import nms
from datasets.factory import get_imdb
from utils.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser(
        description="generate regions from RPN and classify from higher res")
    parser.add_argument('--rpn-def', dest='rpn_def',
                        help='prototxt defining the rpn',
                        default='models/pascal_voc/VGG16/faster_rcnn_alt_opt/'
                        'faster_rcnn_test_onebit_224.pt')
    parser.add_argument('--rpn-net', dest='rpn_net', help='rpn caffemodel',
                        default='output/faster_rcnn_alt_opt/voc_2007_trainval/'
                        'VGG16_faster_rcnn_final_onebit_224.v2.caffemodel')
    parser.add_argument('--cls-def', dest='cls_def',
                        help='prototxt defining the classification network'
                        ' (if not included, will use same prototxt as rpn)',
                        default=None)
    parser.add_argument('--cls-net', dest='cls_net',
                        help='classification network caffemodel (if not '
                        'included, will use same network as rpn, ignoring '
                        'classification prototxt',
                        default=None)
    parser.add_argument('--imdb', dest='imdb', default='voc_2007_test',
                        help='image database')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def vis_roi(im, roi, ij=False):
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    if ij:
        roi = roi[[1, 0, 3, 2]]
    plt.cla()
    plt.imshow(im)
    plt.gca().add_patch(
        plt.Rectangle((roi[0], roi[1]),
                      roi[2] - roi[0],
                      roi[3] - roi[1], fill=False,
                      edgecolor='b', linewidth=3))
    plt.show()


def get_rois(net, im, scale=(224, 224)):
    net_img, im_scale = preprocess_img(im, scale)
    img_blob = im_list_to_blob([net_img])
    kwargs = setup_net_inputs(net, img_blob, im_scale)
    blobs_out = net.forward(end='proposal', **kwargs)

    rois = blobs_out['rois'] / im_scale

    return rois


def clip_roi(roi, shape):
    roi[0] = np.minimum(np.maximum(roi[0], 0), shape[0])
    roi[1] = np.minimum(np.maximum(roi[1], 0), shape[1])
    roi[2] = np.minimum(np.maximum(roi[2], 0), shape[0])
    roi[3] = np.minimum(np.maximum(roi[3], 0), shape[1])


def classify_rois(net, im, rois, scale=(224, 224), vis=False):

    num_rois = rois.shape[0]
    all_boxes = np.zeros((num_rois, 84))
    all_scores = np.zeros((num_rois, 21))
    for i in xrange(num_rois):
        roi = rois[i, :]
        # using ij axis indexing
        roi = roi.astype(int, copy=False)[1:][[1, 0, 3, 2]]
        clip_roi(roi, im.shape)

        shape = [roi[2] - roi[0], roi[3] - roi[1]]

        ax_min = np.argmin(shape)
        ax_max = np.argmax(shape)

        # If the lengths are equal, set one axis to become opposite
        if ax_min == ax_max:
            ax_max = 1 - ax_max
        im_patch = np.zeros((shape[ax_max], shape[ax_max], 3))

        diff = shape[ax_max] - shape[ax_min]
        ext1 = roi[ax_min] - (diff + 1) / 2
        ext2 = roi[ax_min + 2] + diff / 2

        im_ind = [slice(None)] * im.ndim
        patch_ind = [slice(None)] * len(shape)

        im_ind[ax_min] = slice(max(0, ext1), min(im.shape[ax_min], ext2))
        im_ind[ax_max] = slice(roi[ax_max], roi[ax_max + 2])

        patch_ind[ax_min] = slice(max(0, -ext1),
                                  min(shape[ax_max],
                                      shape[ax_max] - (ext2 - im.shape[ax_min]))
                                  )
        patch_ind[ax_max] = slice(0, shape[ax_max])

        im_patch[patch_ind] = im[im_ind].astype(np.float32, copy=True)
        if vis:
            import matplotlib.pyplot as plt
            vis_roi(im, roi, ij=True)
            plt.imshow(im_patch[:, :, (2, 1, 0)] / 255)
            plt.show()

        net_img, im_scale = _preprocess_img(im_patch, scale)
        img_blob = im_list_to_blob([net_img])
        kwargs = setup_net_inputs(net, img_blob, im_scale)

        net.forward(end='proposal', **kwargs)

        # TODO: Instead of giving whole image, use the exact roi
        net.blobs['rois'].reshape(1, 5)
        net.blobs['rois'].data[...] = np.asarray([0, 0, 0, 224, 224])
                                                  # patch_ind[1].start,
                                                  # patch_ind[0].start,
                                                  # patch_ind[1].stop,
                                                  # patch_ind[0].stop])

        blobs_out = net.forward(start='roi_pool5')
        scores, boxes, _ = process_net_outputs(net, blobs_out, net_img.shape)
        boxes /= im_scale
        boxes += np.tile([im_ind[1].start, im_ind[0].start], 42)

        all_boxes[i, :], all_scores[i, :] = boxes, scores

    return all_scores, all_boxes


def roi_cropping_test(rob_net, cls_net, imdb, max_per_image=100, thresh=0.05,
                      vis=True):
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    output_dir = get_output_dir(imdb, None)
    for k in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(k))
        _t = {"stage1": Timer(), "stage2": Timer()}

        _t['stage1'].tic()
        rois = get_rois(rpn_net, im)
        _t['stage1'].toc()

        _t['stage2'].tic()
        scores, boxes = classify_rois(cls_net, im, rois)
        _t['stage2'].toc()

        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]

            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS - 0.1)
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

        print "{}/{} {:.3f}s {:.3f}s".format(k + 1, num_images,
                                             _t["stage1"].average_time,
                                             _t["stage2"].average_time)
    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)


if __name__ == "__main__":
    args = parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)

    pprint.pprint(cfg)

    rpn_net = caffe.Net(args.rpn_def, caffe.TEST, weights=args.rpn_net)
    if args.cls_def and args.cls_net:
        cls_net = caffe.Net(args.cls_def, caffe.TEST, weights=args.cls_net)
    elif args.cls_net:
        cls_net = caffe.Net(args.rpn_def, caffe.TEST, weights=args.cls_net)
    else:
        cls_net = rpn_net

    imdb = get_imdb(args.imdb)

    roi_cropping_test(rpn_net, cls_net, imdb)
