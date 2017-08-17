from easydict import EasyDict
import numpy as np

cfg = EasyDict()

cfg.ROOT = "/home/eriche/"
cfg.IMAGES = cfg.ROOT + "Projects/Release/GTISDK/Bin/Image_bmp/"
# cfg.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
cfg.PIXEL_MEANS = np.array([[[0, 0, 0]]])
cfg.NMS = 0.3
cfg.IMDB = "voc_2007_test"

# caffe parameters
cfg.CAFFE = EasyDict()
# If 1, use CPU - not recommended
cfg.CAFFE.CPU_MODE = 0
# ID of GPU Device
cfg.CAFFE.GPU_ID = 0
# Number of classes to be identified
cfg.CAFFE.NUM_CLASSES = 21
# target length for minimum side
cfg.CAFFE.IM_MIN = 600
# upper bound for maximum side
cfg.CAFFE.IM_MAX = 1000
# Score threshold for which to accept box
cfg.CAFFE.THRESH = 0.05
# Path to caffe prototxt file
cfg.CAFFE.DEF = cfg.ROOT + "py-faster-rcnn/models/pascal_voc/VGG16/" \
                           "faster_rcnn_alt_opt/faster_rcnn_test.pt"
# cfg.CAFFE.DEF = cfg.ROOT + "py-faster-rcnn/models/pascal_voc/VGG_binary/" \
#                            "test_vggbinary.prototxt"
# Path to convolution only prototxt file
cfg.CAFFE.CONV_DEF = cfg.ROOT + "py-faster-rcnn/models/pascal_voc/VGG_binary/" \
                                "faster_rcnn_convolution_binary_test.prototxt"
# Path to rpn and fc only prototxt file
cfg.CAFFE.FC_DEF = cfg.ROOT + "py-faster-rcnn/models/pascal_voc/VGG_binary/" \
                              "faster_rcnn_rpn_fc_test.prototxt"
# Path to caffemodel (weights) file
cfg.CAFFE.CAFFEMODEL = \
    cfg.ROOT + "py-faster-rcnn/output/faster_rcnn_binary/" \
               "vgg16_faster_rcnn_binary_mean0_Aug9.caffemodel"
# Classes for identification - length should be equal to cfg.CAFFE.CLASSES
cfg.CAFFE.CLASSES = ('__background__',  # always index 0
                     'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor')

# fpga parameters
cfg.FPGA = EasyDict()
# Choose to use FTDI
cfg.FPGA.FTDI = 0
cfg.FPGA.DTYPE = cfg.FPGA.FTDI
# Resized width and height for image input o FPGA
cfg.FPGA.GTI_IMAGE_WIDTH = 224
cfg.FPGA.GTI_IMAGE_HEIGHT = 224
# 0 for Normal, 1 for Learning Mode, 2 for feature map extraction before
# last pooling layer
cfg.FPGA.CNNMODE = 2
# Turn last pooling on or off
cfg.FPGA.POOLING = 0
# Size of output from FPGA (512 * 14 * 14)
cfg.FPGA.OUTLEN = 0x18800
# ID of FPGA Device
cfg.FPGA.DEVICE_NAME = '0'
# Coefficients file for FPGA convolution weights
cfg.FPGA.COEF_FILE = "coef_learn_new.dat"
# cfg.FPGA.COEF_FILE = cfg.ROOT + "Projects/Release/GTISDK/Bin/Models/" \
                                # "coef_learn_new.dat"


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if b not in k:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = EasyDict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, cfg)
