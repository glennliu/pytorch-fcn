#!/usr/bin/env python

import argparse
import os
import os.path as osp

import fcn
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import torchfcn
import tqdm
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from mydataLoader import VOC2011ClassSeg_revised

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    root = osp.expanduser('~/data/datasets')
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2011ClassSeg(
            root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

    n_class = len(val_loader.dataset.class_names)

    png_root = osp.expanduser('~/Documents/5421_P1/my_images')
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(360,500),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    my_testset = VOC2011ClassSeg_revised(png_root,split='val', transform=True)
    my_testLoader = torch.utils.data.DataLoader(my_testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # print(len(testloader))

    n_class = len(my_testLoader.dataset.class_names)
    print(my_testLoader.dataset.class_names)

    if osp.basename(model_file).startswith('fcn32s'):
        model = torchfcn.models.FCN32s(n_class=21)
    elif osp.basename(model_file).startswith('fcn16s'):
        model = torchfcn.models.FCN16s(n_class=21)
    elif osp.basename(model_file).startswith('fcn8s'):
        if osp.basename(model_file).startswith('fcn8s-atonce'):
            model = torchfcn.models.FCN8sAtOnce(n_class=21)
        else:
            model = torchfcn.models.FCN8s(n_class=21)
    else:
        raise ValueError
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    print('==> Evaluating with VOC2011ClassSeg seg11valid')
    visualizations = []
    label_trues, label_preds = [], []
    cnt = 0
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(my_testLoader),
                                               total=len(my_testLoader),
                                               ncols=80, leave=False):
        # print(batch_idx)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        score = model(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()

        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = my_testLoader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)
            # if len(visualizations) < 4:
            viz = fcn.utils.visualize_segmentation(
                lbl_pred=lp, img=img, n_class=n_class,
                label_names=my_testLoader.dataset.class_names)
            visualizations.append(viz)
            viz = fcn.utils.get_tile_image(visualizations)
            skimage.io.imsave('./demo_result/viz_evaluate'+str(cnt)+'.png', viz)
            visualizations = []
            cnt = cnt+1

    metrics = torchfcn.utils.label_accuracy_score(
        label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))
    # print('label trues: %s' % label_preds)
    print('lp: %s' % lp)
    print('lt: %s' % lt)
    # print('lable predict: %s' % label_trues)
    # print('img: %s' % len(img))




if __name__ == '__main__':
    main()
