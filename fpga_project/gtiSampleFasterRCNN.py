
import cv2
import numpy as np

from Tkinter import Tk
from tkFileDialog import askopenfilename
from NetInterface import FasterRCNNInterface as FasterRCNN, VocTester
from config import cfg
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser(description="Run original Faster RCNN on gpu and"
                            " fpga")
    parser.add_argument('--gpu', dest='gpu', help='run fc on gpu',
                        default=False, action='store_true')
    parser.add_argument('--fpga', dest='fpga', help='run conv with fpga',
                        default=False, action='store_true')
    parser.add_argument('--test', dest='test', help='run tester',
                        default=False, action='store_true')

    args = parser.parse_args()
    return args


def run_faster_rcnn(im, im_scale):
    if args.fpga:
        blobs_out = net.do_fpga_conv(im)
    else:
        blobs_out = net.do_gpu_conv(im, im_scale)

    # print np.sum(blobs_out1), np.sum(blobs_out2)
    # for i in xrange(5):
    #     plt.figure(num='fpga')
    #     plt.imshow(blobs_out1[0, i, ...])
    #
    #     plt.figure(num='gpu')
    #     plt.imshow(blobs_out2[0, i, ...])
    #     plt.show(block=False)
    #     raw_input("press enter")
    #     plt.close('all')
    scores, boxes = net.do_fc(blobs_out, im.shape, im_scale)

    return scores, boxes


if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        cfg.CAFFE.CPU_MODE = False
    else:
        cfg.CAFFE.CPU_MODE = True

    net = FasterRCNN()
    if args.test:
        VocTester(run_faster_rcnn, reshape=lambda x: FasterRCNN.reshape(x, 224)).run()
    while True:
        root = Tk()
        root.withdraw()

        filename = askopenfilename(parent=root, title="Open File",
                                   initialdir=cfg.IMAGES)
        if type(filename) == tuple:
            raise Exception("filename is tuple")
        img = cv2.imread(filename)
        im, im_scale = FasterRCNN.fpga_reshape(img)
        scores, boxes = run_faster_rcnn(im, im_scale)
        net.show_detections(im, scores, boxes)
