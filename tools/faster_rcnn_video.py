# --------------------------------------------------
# Gyrfalcontech Faster RCNN
# 
# Applies Faster RCNN to a video file. For demo
# purposes only
# 
# Author: Jonathan Zhang, 2017
# --------------------------------------------------
import _init_paths

import caffe
import numpy as np
import cv2
from caffe_utils import process_net_outputs, setup_net_inputs
from caffe_utils import preprocess_img, im_list_to_blob
from fast_rcnn.nms_wrapper import nms


def show_detections(img, cls_name, dets, thresh=0.5):
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 0, 255), 2)
            cv2.putText(img, cls_name, (bbox[0], int(bbox[1] + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)


def run_video_rcnn(model, weight, vfile, thresh=0.05):
    classes = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    net = caffe.Net(model, caffe.TEST, weights=weight)
    vs = cv2.VideoCapture(vfile)
    print vfile
    if not vs.isOpened():
        raise Exception("failed to open video file")
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", 800, 600)
    cv2.moveWindow("Video", 400, 100)
    for frame in xrange(2000):
        print "iter: ", frame
        ret, img_original = vs.read()
        if ret is False:
            continue
        img = img_original.copy()
        im_proc, im_scale = preprocess_img(img, 224, 1000)
        im_blob = im_list_to_blob([im_proc])
        forward_kwargs = setup_net_inputs(net, im_blob, im_scale)
        blobs_out = net.forward(**forward_kwargs)
        scores, boxes, _ = process_net_outputs(net, blobs_out, img.shape,
                                               im_scale)

        for j in xrange(1, 21):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, 0.3)
            cls_dets = cls_dets[keep, :]
            show_detections(img, classes[j], cls_dets)
        # import matplotlib.pyplot as plt
        # plt.imshow(img_original[:, :, (2, 1, 0)])
        # plt.show()
        cv2.imshow('Video', img)
        cv2.waitKey(1)
    vs.release()
    cv2.destroyAllWindows


if __name__ == "__main__":
    vfile = '/home/jonathanzhang/py-faster-rcnn/workspace/All.mp4'
    model = '/home/jonathanzhang/py-faster-rcnn/models/pascal_voc/VGG16/' \
            'faster_rcnn_alt_opt/faster_rcnn_test_imagenet_600.pt'
    weight = '/home/jonathanzhang/py-faster-rcnn/output/' \
             'faster_rcnn_alt_opt/voc_2007_trainval/' \
             'VGG16_faster_rcnn_final_imagenet_.caffemodel'
    caffe.set_mode_gpu()
    caffe.set_device(0)
    run_video_rcnn(model, weight, vfile)
