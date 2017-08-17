import gtilib
import numpy as np
import cv2
import os.path as osp
import sys

from struct import unpack
from config import cfg

def add_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)


caffe_path = osp.join('~', 'py-faster-rcnn', 'caffe-fast-rcnn', 'python')
add_path(osp.expanduser(caffe_path))
import caffe

lib_path = osp.join('~', 'py-faster-rcnn', 'lib')
add_path(osp.expanduser(lib_path))

from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test import vis_detections
from datasets.factory import get_imdb


class FasterRCNNInterface():
    '''Class wrapper for fpga and gpu fasterRCNN demos
    '''
    def __init__(self):
        # gpu setup
        if cfg.CAFFE.CPU_MODE:
            print "Running in CPU mode"
            caffe.set_mode_cpu()
            self.fc = caffe.Net(cfg.CAFFE.FC_DEF, caffe.TEST,
                                weights=cfg.CAFFE.CAFFEMODEL)
            # do not run convolutions on the cpu
            self.conv = None
        else:
            print "Running in GPU mode"
            caffe.set_mode_gpu()
            caffe.set_device(cfg.CAFFE.GPU_ID)
            self.conv = caffe.Net(cfg.CAFFE.CONV_DEF, caffe.TEST,
                                  weights=cfg.CAFFE.CAFFEMODEL)
            self.fc = caffe.Net(cfg.CAFFE.FC_DEF, caffe.TEST,
                                weights=cfg.CAFFE.CAFFEMODEL)

        # fpga setup
        gtilib.gSetType(cfg.FPGA.DTYPE)
        gtilib.gSetCnnMode(cfg.FPGA.CNNMODE)
        gtilib.gSetPooling(cfg.FPGA.POOLING)
        self.handle = gtilib.gOpen(cfg.FPGA.DEVICE_NAME)
        gtilib.gInit(cfg.FPGA.COEF_FILE)

        # voc testing
        self.imdb = get_imdb(cfg.IMDB)

    def do_gpu_conv(self, img, im_scale):
        '''Takes an image and returns the result of the final convolution layer
        after running on the gpu.

        im: preprocessed ndarray image
        im_scale: float used to scale the original image
        '''
        if not self.conv:
            raise Exception("Do not run convolutions on the cpu")
        im = img.astype(np.float32, copy=True)
        im -= cfg.PIXEL_MEANS
        blob = im[np.newaxis, ...]
        channel_swap = (0, 3, 1, 2)
        data = blob.transpose(channel_swap)
        im_info = np.array([[data.shape[2], data.shape[3], im_scale]])

        self.conv.blobs['data'].reshape(*(data.shape))
        self.conv.blobs['im_info'].reshape(*(im_info.shape))
        forward_kwargs = {'data': data.astype(np.float32, copy=False)}
        forward_kwargs["im_info"] = im_info.astype(np.float32, copy=False)
        blobs_out = self.conv.forward(end='relu5_3', **forward_kwargs)
        # print blobs_out['conv5_3']
        return blobs_out['conv5_3']

    def do_fpga_conv(self, img):
        '''Takes and image and returns the result of the final convolution layer
        after running on the fpga.the

        im: image in correct size
        '''
        assert img.shape[0] == cfg.FPGA.GTI_IMAGE_HEIGHT and \
            img.shape[1] == cfg.FPGA.GTI_IMAGE_WIDTH, \
            "incompatble dim: {}".format(img.shape)

        im = img.astype(np.uint8, copy=True)
        b, g, r = cv2.split(im)
        gti_im = np.concatenate((b, g, r))
        outBuff = np.arange(cfg.FPGA.OUTLEN, dtype=np.uint8).tostring()
        # gtilib.gHandleOneFrame(self.handle, gti_im.tostring(), gti_im.size,
        #                        outBuff, cfg.FPGA.OUTLEN)
        # conv_list = [float(unpack('B', c)[0]) for c in outBuff]
        flag, conv_list = gtilib.gHandleOneFrameLt(
            self.handle, gti_im.tostring(), gti_im.size, outBuff,
            cfg.FPGA.OUTLEN)
        blobs_out = np.asarray(conv_list).reshape((512, 14, 14))
        blobs_out = blobs_out[np.newaxis, ...]
        # blobs_out = blobs_out * 4
        # print blobs_out.shape, blobs_out
        return blobs_out

    def setup_fc(self, blob, im_shape, im_scale):
        '''utility function to setup network for rpn and fully connected portion

        im_shape: input shape
        '''
        self.fc.blobs['conv5_3'].reshape(*(blob.shape))
        self.fc.blobs['conv5_3'].data[...] = blob.astype(np.float32,
                                                         copy=False)
        im_info = np.asarray([[im_shape[0], im_shape[1], im_scale]],
                             dtype=np.float32)
        self.fc.blobs['im_info'].reshape(*(im_info.shape))
        self.fc.blobs['im_info'].data[...] = im_info.astype(np.float32,
                                                            copy=False)

    def process_output(self, blobs_out, im_scale, im_shape):
        '''Utility function to aggregate output of network
        '''
        rois = self.fc.blobs['rois'].data.copy()
        boxes = rois[:, 1:5] / im_scale

        scores = blobs_out['cls_prob'].copy()
        bbox_deltas = blobs_out['bbox_pred'].copy()

        im_shape_orig = (im_shape[0] / im_scale, im_shape[1] / im_scale,
                         im_shape[2])
        pred_boxes = bbox_transform_inv(boxes, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape_orig)
        return scores, pred_boxes

    def do_fc(self, blob, im_shape, im_scale, start=None):
        '''Takes a feature map and the image shape and runs the caffe fully
        connected layers to return classificaion and bounding box coordinates.

        blob: result of the conv5_3 layer (1 x 512 x 14 x 14)
        im_shape: shape of resized image
        im_scale: scale used to resize image from original
        '''
        self.setup_fc(blob, im_shape, im_scale)
        out_blob = self.fc.forward(start=start)
        scores, boxes = self.process_output(out_blob, im_scale, im_shape)
        return scores, boxes

    def show_detections(self, im, scores, boxes):
        for j in xrange(1, cfg.CAFFE.NUM_CLASSES):
            inds = np.where(scores[:, j] > cfg.CAFFE.THRESH)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))
            keep = nms(cls_dets, cfg.NMS)
            cls_dets = cls_dets[keep, :]
            # print "displaying detections"
            vis_detections(im, cfg.CAFFE.CLASSES[j], cls_dets)

    def get_blobs(self, layer):
        '''Returns blobs from network or None if not found

        layer: layer string name
        '''

        if self.conv and layer in self.conv.blobs:
            return self.conv.blobs[layer]
        elif layer in self.fc:
            return self.fc.blobs[layer]
        else:
            return None

    def voc_test(self, vis=False):
        '''Testing function used to make sure that configurations are correct
        '''
        def f(im, im_scale):

            blobs_out = self.do_gpu_conv(im, im_scale)
            scores, boxes = self.do_fc(blobs_out, im.shape, im_scale)
            return scores, boxes

        tester = VocTester(f)
        tester.run()

    @staticmethod
    def reshape(im, target=600, max_sz=1000):
        '''Standard reshape method used to preserve aspect ratio
        '''
        im_orig = im.astype(np.float32, copy=True)

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        im_scale = float(target / float(im_size_min))
        if np.round(im_scale * im_size_max) > max_sz:
            im_scale = float(max_sz / float(im_size_max))
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        return im, im_scale

    @staticmethod
    def fpga_reshape(img):
        '''Reshapes image to fit dimensions required by the FPGA as specified
        in config. Assumes a square region - if not square, removes the least
        amount from both ends of the longer side to get square.

        im: input image
        '''
        im = img.copy()

        if im.shape[0] != im.shape[1]:
            ax_min = np.argmin(im.shape[:2])
            ax_max = np.argmax(im.shape[:2])
            diff = im.shape[ax_max] - im.shape[ax_min]
            ind = [slice(None)] * im.ndim
            ind[ax_max] = slice((diff + 1) / 2, - diff / 2)
            im = im[ind]

        assert (im.shape[0] == im.shape[1]), im.shape
        im_scale = im.shape[0] / cfg.FPGA.GTI_IMAGE_WIDTH
        im = cv2.resize(im, (cfg.FPGA.GTI_IMAGE_WIDTH,
                             cfg.FPGA.GTI_IMAGE_HEIGHT))
        return im, im_scale


class VocTester():
    def __init__(self, f, reshape=FasterRCNNInterface.reshape,
                 imdb='voc_2007_test'):
        self.f = f
        self.reshape = reshape
        self.imdb = get_imdb(imdb)

    def evaluate(self, scores, boxes, i, all_boxes):
        for j in xrange(1, self.imdb.num_classes):
            inds = np.where(scores[:, j] > cfg.CAFFE.THRESH)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.NMS)
            cls_dets = cls_dets[keep, :]

            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        max_per_image = 100

        image_scores = np.hstack(
            [all_boxes[j][i][:, -1]
             for j in xrange(1, self.imdb.num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in xrange(1, self.imdb.num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]

    def run(self):
        num_images = len(self.imdb.image_index)
        all_boxes = [[[] for i in xrange(num_images)]
                     for j in xrange(self.imdb.num_classes)]
        for i in xrange(num_images):
            print "{}/{}".format(i + 1, num_images)
            img = cv2.imread(self.imdb.image_path_at(i))
            scores, boxes = self.f(*(self.reshape(img)))
            self.evaluate(scores, boxes, i, all_boxes)
        self.imdb.evaluate_detections(all_boxes, ".output")
