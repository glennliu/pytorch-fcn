# pytorch-fcn

[![PyPI Version](https://img.shields.io/pypi/v/torchfcn.svg)](https://pypi.python.org/pypi/torchfcn)
[![Python Versions](https://img.shields.io/pypi/pyversions/torchfcn.svg)](https://pypi.org/project/torchfcn)
[![Build Status](https://travis-ci.org/wkentaro/pytorch-fcn.svg?branch=master)](https://travis-ci.org/wkentaro/pytorch-fcn)

PyTorch implementation of [Fully Convolutional Networks](https://github.com/shelhamer/fcn.berkeleyvision.org).
Based on previous model, a demo program is developed for HKUST COMP5421 Project 1 (Semantic Segmentation)

## Requirements

- [pytorch](https://github.com/pytorch/pytorch) >= 0.2.0
- [torchvision](https://github.com/pytorch/vision) >= 0.1.8
- [fcn](https://github.com/wkentaro/fcn) >= 6.1.5
- [Pillow](https://github.com/python-pillow/Pillow)
- [scipy](https://github.com/scipy/scipy)
- [tqdm](https://github.com/tqdm/tqdm)


## Installation

```bash
git clone https://github.com/wkentaro/pytorch-fcn.git
cd pytorch-fcn
pip install .

# or

pip install torchfcn
```

## Download Trained Model or Training a Model
A model is trained based on test dataset from HKUST COMP5421. And the model can be downloaded [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/cliuci_connect_ust_hk/EYcmHYk52fRJmd01G_6UKmABLeSeVfgbYIt_0VMsOfFUDw?e=s72NPd).
For Training a model by yourself
```bash
cd examples/voc
python train_fcn32s_demo.py
```

## Run demo on validation and test dataset
Download the project1 folder "comp5421_TASK2", which contains folder `train`, `validation` and `test`. And open `examples/voc/demo.py`, change its `pkg_root` to the downloaded `comp5421_TASK2` directory.
Then, 
```bash
cd examples/voc
python demo.py $PYTORCH_MODEL_PATH
```
The program will generate segmention images at `./test_result` and `./val_result`. The terminal also display key accuracy data
```js
lch@gs:~/code_ws/comp5421/pytorch-fcn/examples/voc$ python demo.py logs/20190403_155952.262577/model_best.pth.tar 
/home/lch/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
==> Loading FCN32s model file: logs/20190403_155952.262577/model_best.pth.tar
==> Evaluating with VOC2011ClassSeg seg11valid
  0%|                                                   | 0/150 [00:00<?, ?it/s]demo.py:91: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  data, target = Variable(data, volatile=True), Variable(target)
  0%|                                                    | 0/50 [00:00<?, ?it/s]demo.py:123: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  data, target = Variable(data, volatile=True), Variable(target)
Accuracy: 78.4032333198672                                                      
Accuracy Class: 68.42654277275444
Mean IU: 57.14322250091958
FWAV Accuracy: 66.66971237667332
```


## Other Training
For other training models, see [VOC example](examples/voc).


## Accuracy

At `10fdec9`.

| Model | Implementation |   epoch |   iteration | Mean IU | Pretrained Model |
|:-----:|:--------------:|:-------:|:-----------:|:-------:|:----------------:|
|FCN32s      | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn32s)       | - | -     | **63.63** | [Download](https://github.com/wkentaro/pytorch-fcn/blob/63bc2c5bf02633f08d0847bb2dbd0b2f90034837/torchfcn/models/fcn32s.py#L31-L37) |
|FCN32s      | Ours                                                                                         |11 | 96000 | 62.84 | |
|FCN16s      | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn16s)       | - | -     | **65.01** | [Download](https://github.com/wkentaro/pytorch-fcn/blob/63bc2c5bf02633f08d0847bb2dbd0b2f90034837/torchfcn/models/fcn16s.py#L14-L20) |
|FCN16s      | Ours                                                                                         |11 | 96000 | 64.91 | |
|FCN8s       | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s)        | - | -     | **65.51** | [Download](https://github.com/wkentaro/pytorch-fcn/blob/63bc2c5bf02633f08d0847bb2dbd0b2f90034837/torchfcn/models/fcn8s.py#L14-L20) |
|FCN8s       | Ours                                                                                         | 7 | 60000 | 65.49 | |
|FCN8sAtOnce | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s-atonce) | - | -     | **65.40** | [Download](https://github.com/wkentaro/pytorch-fcn/blob/63bc2c5bf02633f08d0847bb2dbd0b2f90034837/torchfcn/models/fcn8s.py#L177-L183) |
|FCN8sAtOnce | Ours                                                                                         |11 | 96000 | 64.74 | |

<img src=".readme/fcn8s_iter28000.jpg" width="50%" />
Visualization of validation result of FCN8s.
