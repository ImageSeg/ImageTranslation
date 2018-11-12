from __future__ import print_function
import argparse
import os
from math import log10

import numpy as np
from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from networks import define_G, define_D, GANLoss, print_network
from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', default='cityscapes', help='cityscapes or spherical or other dataset name')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--img_w', type=int, default=512, help='image width eg 1024')
parser.add_argument('--img_h', type=int, default=512, help='image height')
parser.add_argument('--xy_size', type=int, default=2, help='location map size')
parser.add_argument('--mode', default="AtoB", help='Traing mode: AtoB or BtoA')
parser.add_argument('--norm', default="linear", help='normalization type of coordconv: linear, c_sin, cycle, c_linear, sigmoid, and fair space if not add -no eg. linear otherwise linear-no')
#parser.add_argument('--goon', action='store_true', help='use goon?')
parser.add_argument('--goon', type=int, default=0, help='epoch to restart training with')
opt = parser.parse_args()

print(opt)
np.save("checkpoint/{}/parameters_train_{}".format(opt.dataset, strftime("%H%M", gmtime())), opt)
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(root_path + opt.dataset, mode=opt.mode, xy=opt.xy_size, img_size=(opt.img_w, opt.img_h), norm=opt.norm)
test_set = get_test_set(root_path + opt.dataset, mode = opt.mode, xy=opt.xy_size, img_size=(opt.img_w, opt.img_h), norm=opt.norm)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.xy_size, 'batch', False, [0])
netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'batch', False, [0])


if opt.goon!=0:
    #Start mith on-the-last-iteration saved model
    netG = torch.load("checkpoint/cityscapes/netG_model_epoch_{}.pth".format(opt.goon))
    netD = torch.load("checkpoint/cityscapes/netD_model_epoch_{}.pth".format(opt.goon))

criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

print('---------- Networks initialized -------------')
#print_network(netG)
#print_network(netD)
print('-----------------------------------------------')

real_a = torch.FloatTensor(opt.batchSize, opt.input_nc, opt.img_h, opt.img_w)
real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, opt.img_h, opt.img_w)

if opt.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()

real_a = Variable(real_a)
real_b = Variable(real_b)


loss_list= np.array([[0, 0.0, len(training_data_loader), 0.0, 0.0]])
def train(epoch):
    loss_list= np.array([[epoch, 0.0, len(training_data_loader), 0.0, 0.0]])
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        fake_b = netG(real_a)

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################

        optimizerD.zero_grad()

        # train with fake
        real_a_no_xy = real_a[:,0:3, ...]
        real_b_no_xy = real_b[:,0:3, ...]
        fake_ab = torch.cat((real_a_no_xy, fake_b), 1)
        pred_fake = netD.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a_no_xy, real_b_no_xy), 1)
        pred_real = netD.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)

        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()

        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a_no_xy, fake_b), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

         # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b_no_xy) * opt.lamb

        loss_g = loss_g_gan + loss_g_l1

        loss_g.backward()

        optimizerG.step()

        loss_list = np.append(loss_list, [[epoch,  iteration, len(training_data_loader), loss_d.data[0], loss_g.data[0]]], axis=0)
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.data[0], loss_g.data[0]))
    #np.save("checkpoint/{}/learning_{}".format(opt.dataset, strftime("%H%M", gmtime())), loss_list)
    np.save("checkpoint/{}/learning_{}".format(opt.dataset, epoch), loss_list)


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0], volatile=True), Variable(batch[1][:,0:opt.input_nc, ...], volatile=True)	#with torch.no_grad():
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = netG(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    np.append(loss_list, [[epoch,  0, 0, avg_psnr / len(testing_data_loader), avg_psnr / len(testing_data_loader)]], axis=0)
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))
    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
    net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
    torch.save(netG, net_g_model_out_path)
    torch.save(netD, net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint/" + opt.dataset))

#if opt.goon and opt.go_epoch==1:
#    opt.go_epoch=200
for epoch in range(opt.goon, opt.nEpochs + 1):
    train(epoch)
    test()
    if epoch % 10 == 0:
        checkpoint(epoch)

