# pix2pix-pytorch

PyTorch implementation of [Image-to-Image Translation Using Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf).

Based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

The examples from the paper: 

<img src="examples.jpg" width = "766" height = "282" alt="examples" align=center />

## Prerequisites

+ Linux
+ Python with numpy
+ NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
+ pytorch
+ torchvision

## Getting Started

+ Clone this repo:

    git clone git@github.com:mrzhu-cool/pix2pix-pytorch.git
    cd pix2pix-pytorch

+ Get dataset

    unzip dataset/facades.zip

+ Train the model:

    python train.py --dataset facades --nEpochs 200 --cuda

+ Test the model:

    python test.py --dataset facades --model checkpoint/facades/netG_model_epoch_200.pth --cuda

## Acknowledgments

This code is a concise implementation of [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Much easier to understand.

Highly recommend the more completed and organized code [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by original author junyanz.



Run:
nshimyimana@serv-2103:/novelview/dominique/pix2pix-pytorch_locationsxy$ CUDA_VISIBLE_DEVICES=4 python3.6 validate.py --dataset spherical --save


Test:
nshimyimana@serv-2103:/novelview/dominique/pix2pix-pytorch_locationsxy$ python3.6 test.py

nshimyimana@serv-2103:/novelview/dominique/pix2pix-pytorch_locationsxy$ CUDA_VISIBLE_DEVICES=3 python3.6 test.py --dataset cityscapes --model checkpoint/cityscapes/netG_model_epoch_200.pth --cuda --img_size 512 512

nshimyimana@serv-2103:/novelview/dominique/pix2pix-pytorch_locationsxy$ CUDA_VISIBLE_DEVICES=3 python3.6 test.py --dataset cityscapes --model checkpoint/cityscapes/netG_model_epoch_200.pth --cuda


Train/retrain:
nshimyimana@serv-2103:/novelview/dominique/pix2pix-pytorch_locationsxy$ CUDA_VISIBLE_DEVISES=0 python3.6 train.py --goon --cuda --dataset spherical

nshimyimana@serv-2103:/novelview/dominique/pix2pix-pytorch_locationsxy$ CUDA_VISIBLE_DEVISES=5 python3.6 train.py --xy_size 0 --cuda --dataset spherical

