# --------------------------------------------------------
# ROI Generator R-CNN (variant of Faster RCNN by Ross Girshick)
# 
# 224 x 224 fixed size input network
# Generates regions using RPN, crops regions from original image, and classifies
# and performs bounding box regression on image regions.
# Written by Jonathan Zhang
# --------------------------------------------------------

import cv2
import numpy as np
import sys

from Tkinter import Tk
from tkFileDialog import askopenfilename
from NetInterface import FasterRCNNNetInterface, VocTester
from config import cfg
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Run original Faster RCNN on gpu and"
                            " fpga")
    parser.add_argument('--gpu', dest='gpu', help='run fc on gpu',
                        default=False, action='store_true')
    parser.add_argument('--fpga', dest='fpga', help='run conv with fpga',
                        default=False, action='store_true')

    args = parser.parse_args()
    return args


def reshape_img(img):
    im = img.astype(np.float32, copy=True)
    im -= cfg.PIXEL_MEANS
    if im.shape[0] != im.shape[1]:
        ax_min = np.argmin(im.shape[:2])
        ax_max = np.argmax(im.shape[:2])
        diff = im.shape[ax_max] - im.shape[ax_min]
        ind = [slice(None)] * im.ndim
        ind[ax_min] = slice((diff + 1) / 2, diff / 2)
        im = im[ind]

    assert (im.shape[0] == im.shape[1])
    im_scale = im.shape[0] / cfg.FPGA.GTI_IMAGE_WIDTH
    im = cv2.resize(im, (cfg.FPGA.GTI_IMAGE_WIDTH, cfg.FPGA.GTI_IMAGE_HEIGHT))
    return im, im_scale


def clip_roi(roi, shape):
    roi[0] = np.minimum(np.maximum(roi[0], 0), shape[0])
    roi[1] = np.minimum(np.maximum(roi[1], 0), shape[1])
    roi[2] = np.minimum(np.maximum(roi[2], 0), shape[0])
    roi[3] = np.minimum(np.maximum(roi[3], 0), shape[1])


def get_region_from_image(im, roi):
    '''Crops the specified region from the original image
    '''
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

    return im_patch, im_ind[1].start, im_ind[0].start


def classify_rois(net, im, rois, scale=(224, 224)):
    num_rois = rois.shape[0]
    all_boxes = np.zeros((num_rois, 84))
    all_scores = np.zeros((num_rois, 21))
    for i in xrange(num_rois):
        im_patch, dx, dy = get_region_from_image(im, rois[i])
        assert im_patch.shape[0] == im_patch.shape[1]
        im_scale = scale[0] / np.max(im_patch.shape)
        im_patch = cv2.resize(im_patch, scale)
        # rois must be same scale as transformed images
        conv5_3 = net.do_fpga_conv(im_patch)
        net.setup_fc(conv5_3, scale, im_scale)

        net.fc.forward(end='proposal')
        rois = np.asarray([0, 0, 0, 224, 224])
        net.get_blobs('rois').reshape(*(rois.shape))
        net.get_blobs('rois').data[...] = rois

        # This has to be removed from outside the class
        blobs_out = net.fc.forward(start='roi_pool5')
        scores, boxes = net.process_output(blobs_out, im_scale, (224, 224, 3))

        boxes += np.tile([dx, dy], 42)
        # print boxes
        # print "boxes score", boxes.shape, scores.shape
        all_boxes[i, :], all_scores[i, :] = boxes, scores

    return all_scores, all_boxes


def run_roi_gen_rcnn(im, im_scale):
    if args.fpga:
        conv5_3 = net.do_fpga_conv(im)
    else:
        conv5_3 = net.do_gpu_conv(im, im_scale)
    net.do_fc(conv5_3, im.shape, im_scale)
    rois = net.get('rois').data / im_scale
    scores, boxes = classify_rois(net, im, rois)
    return scores, boxes


if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        cfg.CAFFE.CPU_MODE = False
    else:
        cfg.CAFFE.CPU_MODE = True

    net = FasterRCNNNetInterface()
    if args.test:
        VocTester(run_roi_gen_rcnn).run()
    while True:
        root = Tk()
        root.withdraw()

        filename = askopenfilename(parent=root, title="Open File",
                                   initialdir=cfg.IMAGES)
        if type(filename) == tuple:
            raise Exception("filename is tuple")
        img = cv2.imread(filename)
        im, im_scale = reshape_img(img)

        scores, boxes = run_roi_gen_rcnn(im, im_scale)
        # get rois
        net.show_detections(img, scores, boxes)
