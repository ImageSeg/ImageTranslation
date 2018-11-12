from __future__ import print_function
import argparse
from time import gmtime, strftime
import os
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from util import is_image_file, load_img, save_img, diagnose_network, classify, decode_ids, save_img_np, encode_img_ids, rm_2channel

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', default='cityscapes', help='cityscapes, spherical')
parser.add_argument('--model', type=str, default='default', help='model file to use e.g. checkpoint/cityscapes/netG_model_epoch_250.pth or use --dataset norm and --it iteration')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--img_size', nargs='+',type=int, default=(512,512), help='image size as tupel z.B --img_size 1024 512')
parser.add_argument('--xy_size', type=int, default=2, help='0 without location map')
parser.add_argument('--it', type=int, default=250, help='model iteration to get model name (name_iteation)')
parser.add_argument('--rotated', type=int, default=0, help='90, 180, 270 test on rotation')
parser.add_argument('--norm', type=str, default='linear', help='0 (without location map), c_sin or cycle')
parser.add_argument('--m', type=str, default='fair', help='0 without fair space')
opt = parser.parse_args()

if opt.model=="default":
    opt.model="checkpoint/{}/linear/netG_model_epoch_250.pth".format(opt.dataset)
else: opt.model="checkpoint/{}/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.norm, opt.it)
print(opt)

#np.save("result/{}/parameters_test_{}".format(opt.dataset, strftime("%H%M", gmtime())), opt)

rgb = False if opt.xy_size > 0 else True
netG = torch.load(opt.model)
#diagnose_network(netG)

image_dir = "dataset/{}/test/a/".format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img1 = load_img(image_dir + image_name, opt.img_size, norm=opt.norm, rgb=rgb, rotated=opt.rotated)
    img = transform(img1)
    #print(np.unique(img))
    input = Variable(img, volatile=True).view(1, -1, opt.img_size[1], opt.img_size[0])

    if opt.cuda:
        netG = netG.cuda()
        input = input.cuda()

    out = netG(input)
    out = out.cpu()
    out_img = out.data[0]
    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.mkdir(os.path.join("result", opt.dataset))
    save_img(out_img, "result/{}/t_{}".format(opt.dataset, image_name), opt.img_size)
    save_img_np(rm_2channel(img1, sz=opt.img_size), "result/{}/o_{}".format(opt.dataset, image_name), opt.img_size)

#This classification(classify,encode_img_ids) will be updated to run on gpu
    if opt.m=='0':
        save_img_np(decode_ids(classify(out_img, m=opt.m), m=opt.m), "result/{}/{}".format(opt.dataset, image_name), opt.img_size) #quadtree
    else:
        save_img_np(decode_ids(encode_img_ids(out_img)), "result/{}/{}".format(opt.dataset, image_name), opt.img_size)
