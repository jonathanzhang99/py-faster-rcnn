# Documentation

Refer to the official Faster R-CNN README.md (titled `README.md`) for installation instructions.
Some notes for Caffe installation in no particular order
- If using CUDA 8.0 or CUDNN 5, the included caffe base will be incompatible. You will have to fetch and merge the current Caffe version. 
- If using CPU only version of Faster RCNN, please refer [here](TODO://linkToCPUInstructions)

### Contents
1. [Training](#training)
2. [Testing](#testing)
3. [Experiments](#experiments)
	1. [ROI Substitution Test](#roi-substitution-test)
	2. [ROI Augmentation Test](#roi-augmentation-test)
	3. [ROI Cropping Test](#roi-cropping-test)
	4. [Tile RCNN](#tile-rcnn)


### Training

Refer to the official documentation for training hardware requirements. All code examples are done in the root directory of this respository. We opt for end-to-end training in all examples as it is faster and produces extremely similar accuracy.

1. Normal Training
```Shell
# assume that your GPU id is 0 
# dataset is either pascal_voc or coco
./experiments/scripts/faster_rcnn_end2end.sh 0 VGG16 {dataset}
```
2. Training for 224x224 images
```shell
./experiments/scripts/faster_rcnn_end2end_224x224.sh 0 VGG16 {dataset}
```

**Notes:**
- In order to train a simulated FPGA model, the mean must be set to 0. The only way to do this is to modify `lib/fast_rcnn/config.py`. In that file, change:
```python
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
```
to
```python
__C.PIXEL_MEANS = np.array([[[0, 0, 0, ]]])
```
- If you want to make any other custom adjustments to the configurations, create a new .yaml file in `experiments/cfgs/`
- If you want to use your own network, as opposed to the default VGG16 Imagenet model, you will have to create your own training script and change `--weights data/imagenet_models/${NET}.v2.caffemodel \` to point to your own caffemodel


### Testing
1. Normal testing for VGG16 network
```Shell
./tools/test_net.py \
	--gpu 0 \
	--def models/pascal_voc/VGG16/faster_rcnn_end2end/faster_rcnn_test.py \
	--net output/faster_rcnn_end2end/voc_2007_trainval/{TRAINED_NETWORK} \
	--cfg experiments/cfgs/{YOUR_CFG_FILE} \
	--imdb voc_2007_test
```

### Experiments

The main bottleneck using the FPGA architecture results from the fixed 224x224x3 input size to the network. Decreasing image resolution drastically decreases accuracy of predictions and in this case, we find that using 224x224 rather than the original 600x1000 images decreases mean Average Precision (mAP) by approximately 20 percentage points (from 70% mAP to 50% mAP). The following experiments try to interpret and find solutions to lower this discrepancy.

#### Roi Substitution Test
**File:** `tools/roi_substitute_faster_rcnn.py`
1. Takes two networks, a region proposer and a classifier network
2. Generates ROIs from the region proposer
3. Feeds ROIs directly into the classifier network, clobbering the network's
original ROIs
4. Performs evaluation

**Purpose:** Able to test the difference in
1. quality between ROIs generated from a higher resolution image vs that of a lower resolutino
2. classification/bounding box ability between a higher resolution feature map and a lower resolution feature map given the same ROIs


#### Roi Augmentation Test
File: `tools/roi_augmented_faster_rcnn.py`
1. Takes the original image and slides a window across
