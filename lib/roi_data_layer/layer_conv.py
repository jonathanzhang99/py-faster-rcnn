# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
import math
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
from roi_data_layer.layer import BlobFetcher
import numpy as np
import yaml
from multiprocessing import Process, Queue

class ConvDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    @staticmethod
    def ceil(x):
        return int(math.ceil(x))

    def generate_conv5_3(self, data, window_size=224, stride=128):
        conv_final = np.zeros((1, 512, self.ceil(data.shape[2] / 16.0),
                               self.ceil(data.shape[3] / 16.0)))

        im_height = data.shape[2]
        im_width = data.shape[3]

        # Need to find ceiling of integer division
        num_height_windows = (im_height - window_size + stride - 1) / stride + 1
        num_width_windows = (im_width - window_size + stride - 1) / stride + 1

        for j in xrange(num_height_windows):
            h_stride = min(j * stride, im_height - window_size)
            for i in xrange(num_width_windows):
                w_stride = (min(i * stride, im_width - window_size))
                # print w_stride

                window_blob = {}

                window_blob['data'] = data[:, :,
                                           h_stride: h_stride + window_size,
                                           w_stride: w_stride + window_size]
                assert window_blob['data'].shape == (1, 3, 224, 224), \
                    (h_stride, w_stride, data.shape)
                # plt.imshow((window_blob['data'][0, ...].transpose((1, 2, 0)) +
                # cfg.PIXEL_MEANS)[:, :, (2, 0, 1)] / 255) plt.show()
                self.net.blobs['data'].reshape(*(window_blob['data'].shape))

                forward_kwargs = {}
                forward_kwargs['data'] = window_blob['data'].astype(
                    np.float32, copy=True)

                blobs_out = self.net.forward(**forward_kwargs)

                dx = window_size / 16
                h0 = h_stride / 16
                w0 = w_stride / 16
                map_slice = conv_final[:, :, h0:h0 + dx, w0:w0 + dx]
                assert blobs_out['conv5_3'].shape[2:] == (dx, dx), \
                    (blobs_out['conv5_3'].shape, window_blob['data'].shape)
                # print blobs_out['conv5_3'][0, test, ...]
                # plt.imshow(blobs_out['conv5_3'][0, 3, ...])
                # plt.show()
                np.maximum(map_slice, blobs_out['conv5_3'], out=map_slice)

        return conv_final

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        self.net = caffe.Net(layer_params['prototxt'], caffe.TEST,
                             weights=layer_params['weights'])
        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
                         max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 4)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1

            top[idx].reshape(1, 512, self.ceil(max(cfg.TRAIN.SCALES) / 16.0),
                             self.ceil(cfg.TRAIN.MAX_SIZE / 16.0))
            self._name_to_top_map['conv5_3'] = idx
            idx += 1

        else: # not using RPN
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            top[idx].reshape(1, 5)
            self._name_to_top_map['rois'] = idx
            idx += 1

            # labels blob: R categorical labels in [0, ..., K] for K foreground
            # classes plus background
            top[idx].reshape(1)
            self._name_to_top_map['labels'] = idx
            idx += 1

            if cfg.TRAIN.BBOX_REG:
                # bbox_targets blob: R bounding-box regression targets with 4
                # targets per class
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_targets'] = idx
                idx += 1

                # bbox_inside_weights blob: At most 4 targets per roi are active;
                # thisbinary vector sepcifies the subset of active targets
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_inside_weights'] = idx
                idx += 1

                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_outside_weights'] = idx
                idx += 1

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        blobs['conv5_3'] = self.generate_conv5_3(blobs['data'])
        assert blobs['conv5_3'].shape[0:2] == (1, 512)
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
