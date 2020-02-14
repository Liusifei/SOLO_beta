# SOLO unofficial
The code is an unofficial pytorch implementation of [SOLO: Segmenting Objects by Locations](https://arxiv.org/abs/1912.04488)


## Install
The code is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please check [Install.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md) for installation instructions.

## Training 
### Basic
Follows the same way as mmdetection.
single GPU: (for pdb debug)
```python tools/train.py configs/solo/r50_p2_p6_match_ml.py --work_dir r50_p2_p6_match_ml```
multi GPU (for example 8): 
```./tools/dist_train.sh configs/solo/r50_p2_p6_match_mlaug.py 8 --work_dir r50_p2_p6_match_ml_3X```

## Notes

The code only implements the simplest version of SOLO:
* 1x: ```r50_p2_p6_match_ml.py```
* 3x: ```r50_p2_p6_match_ml_aug.py```
* using vanilla SOLO instead of Decoupled SOLO
* implemented the simplest mask-nms: as the authors did not describe it in detail in the paper, the implemented nms is slow,  will improve it in the future.
* **still in progress**

## Results

After training 6 epoches on the coco dataset using the resnet-50 backbone, the AP is 0.091 on val2017 dataset:

![](AP.jpg)

Both good and bad results:

![](solo.jpg)



<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNjQwODE1NjhdfQ==
-->