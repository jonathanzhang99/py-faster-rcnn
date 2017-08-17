
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from Tkinter import Tk
from tkFileDialog import askopenfilename
from NetInterface import FasterRCNNInterface as FasterRCNN, VocTester
from config import cfg
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Run original Faster RCNN on gpu and"
                            " fpga")
    parser.add_argument('--gpu', dest='gpu', help='run fc on gpu',
                        default=False, action='store_true')
    parser.add_argument('--fpga', dest='fpga', help='run conv with fpga',
                        default=False, action='store_true')
    parser.add_argument('--test', dest='test', help='run test on voc',
                        default=False, action='store_true')
    parser.add_argument('--stride', dest='stride',
                        help='pixels between tiles', default=16)

    args = parser.parse_args()
    return args


def ceil(x):
    '''Sanity sake'''
    return int(math.ceil(x))


def run_tile_rcnn(img, im_scale):
    fms = []
    im = img.astype(np.float32, copy=True)
    # plt.imshow(im[:, :, (2, 1, 0)] / 255)
    # plt.show()
    conv_final = np.zeros((1, 512, ceil(im.shape[0] / 16.0),
                           ceil(im.shape[1] / 16.0)))
    conv_blobs = net.do_gpu_conv(im.copy(), im_scale).copy()

    height = im.shape[0]
    width = im.shape[1]

    # width == height for current iteration of FPGA
    window_sz = cfg.FPGA.GTI_IMAGE_WIDTH
    stride = float(args.stride)

    num_height_windows = ceil((height - window_sz) / stride) + 1
    num_width_windows = ceil((width - window_sz) / stride) + 1
    print "windows: {}".format(num_height_windows * num_width_windows)
    for i in xrange(num_height_windows):
        delta_h = int(min(i * stride, height - window_sz))
        for j in xrange(num_width_windows):
            delta_w = int(min(j * stride, width - window_sz))

            tile = im[delta_h:delta_h + window_sz,
                      delta_w:delta_w + window_sz].astype(np.float32, copy=True)
            # plt.imshow(tile[:, :, (2, 1, 0)] / 255)
            # plt.show()
            if args.fpga:
                conv5_3 = net.do_fpga_conv(tile)
            else:
                conv5_3 = net.do_gpu_conv(tile, im_scale)
            dx = window_sz / 16
            h0, w0 = delta_h / 16, delta_w / 16
            map_slice = conv_final[:, :, h0:h0 + dx, w0:w0 + dx]
            assert conv5_3.shape[2:] == (dx, dx)
            np.maximum(map_slice, conv5_3, out=map_slice)
            fms.append(conv5_3)

    for i in xrange(0):
        print i
        print "b1 max: {}".format(np.max(conv_blobs[0, i, ...]))
        print "conv:   {}".format(np.max(conv_final[0, i, ...]))
        plt.figure(num="original " + str(tile.shape))
        plt.imshow(conv_blobs[0, i, ...])
        for j, t in enumerate(fms):
            if j == 0:
                break
            plt.figure(num="map: " + str(j))
            plt.imshow(t[0, i, ...])
        plt.figure(num='artificial conv')
        plt.imshow(conv_final[0, i, ...])

    # print "conv final ", np.sum(conv_final)
    scores, boxes = net.do_fc(conv_final.astype(np.float32, copy=False),
                              im.shape, im_scale)
    # print
    # print net.fc.blobs['im_info'].data
    # print im.shape, im_scale
    # print np.sum(scores), np.sum(boxes)
    # print "scores: ", scores[0, 0:10]
    # print "boxes: ", boxes[0, 0:10]
    # plt.show()
    return scores, boxes


if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        cfg.CAFFE.CPU_MODE = False
    else:
        cfg.CAFFE.CPU_MODE = True

    net = FasterRCNN()
    if args.test:
        VocTester(run_tile_rcnn,
                  reshape=lambda x: FasterRCNN.reshape(x, 224, 1000)).run()
    while True:
        root = Tk()
        root.withdraw()

        filename = askopenfilename(parent=root, title="Open File",
                                   initialdir=cfg.IMAGES)
        if type(filename) == tuple:
            raise Exception("filename is tuple")
        img = cv2.imread(filename)
        im_orig, im_scale = FasterRCNN.reshape(img, 224, 1000)
        scores, boxes = run_tile_rcnn(im_orig, im_scale)
        net.show_detections(img, scores, boxes)
