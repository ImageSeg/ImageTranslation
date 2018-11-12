import torch
import argparse
import numpy as np
import torch.nn as nn

import os

from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from data import get_evaluation_set
from util import scores, save_val, encode_ĺbl_ids, encode_img_ids, encode_img_ids_2, decode_ids, diagnose_network, trans_classe, classify

def validate(args):
    
##################
    norm_map = args.norm.partition('-')[0]
    if 'no' == args.norm.partition('-')[2]:
       m_uniform ='0'
    else:
       m_uniform ='fair'

    if args.model=="default":
        if args.xy_size==0:
            args.model="checkpoint/{}/no_xy/netG_model_epoch_250.pth".format(args.dataset)
            loc="no_xy/"
        elif args.xy_size==2 and norm_map=='cycle':
            args.model="checkpoint/{}/rot_xy/netG_model_epoch_250.pth".format(args.dataset)
            loc = "rot_xy/"
        else:
            args.model="checkpoint/{}/netG_model_epoch_250.pth".format(args.dataset)            
            loc = "{}/".format(args.norm)
    elif args.model=="best":
        args.model="best_model/netG_model_epoch_200_3D+2D*5D.pth"
        loc = 'best'
    elif os.path.exists(args.model):
        True #args.model = args.model
    else:
        args.model="checkpoint/{}/{}/netG_model_epoch_{}.pth".format(args.dataset, norm_map, args.it)
        if not os.path.exists(args.model):
            args.model="checkpoint/{}/pix2pix/netG_model_epoch_{}.pth".format(args.dataset, args.it)
        loc=args.norm
    img_size = (args.img_rows, args.img_cols)
    val_dir = "dataset/{}/val/{}val_image_{}/".format(args.dataset, loc, args.rotated)
    if args.xy_size>0 and norm_map=='':
        raise RuntimeError('Please choose norm parameter for coordconv e.g. linear, cycle, sigmoid')
##################
    print(args)
    np.save("dataset/{}/parameters_val_{}_{}_{}".format(args.dataset, args.xy_size, m_uniform, args.it), args)
    print('===> Loading datasets')
    root_path = "dataset/"
    evaluation_set = get_evaluation_set(root_path + args.dataset, mode = args.mode, xy=args.xy_size, img_size=(args.img_cols, args.img_rows), norm=args.norm, rotated=args.rotated, val=True)
    evaluation_data_loader = DataLoader(dataset=evaluation_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=False)

#####################

    # Setup Model
    netG_model = torch.load(args.model)
    #netG_model.eval()
    if torch.cuda.is_available():
        netG_model = netG_model.cuda()
    #diagnose_network(netG_model)

####################
    gts, preds = [], []
    for i, (images, labels) in tqdm(enumerate(evaluation_data_loader)):
        if torch.cuda.is_available():
            images = Variable(images.cuda(0), volatile=True)
            #labels = Variable(labels.cuda(0), volatile=True)
        else:
            images = Variable(images, volatile=True)
            #labels = Variable(labels, volatile=True)


        with torch.no_grad():
            outputs = netG_model(images)
            outputs = outputs.cpu()
            out_img = outputs.data[0]
            #print(np.unique(out_img), 'out')
            if m_uniform!='0':
                pred = encode_img_ids(out_img)
            else:
                pred = classify(outputs.cpu().data[0], m=m_uniform)	#it is realy slow
            #pred = trans_classe(out_img)
            gt = encode_ĺbl_ids(labels[0].numpy(), m=m_uniform)	#labels.data.cpu().numpy()	# 
            ##gt = classify(labels[0], img=False)       
	
        if args.save:
            save_val(decode_ids(pred, m=m_uniform), decode_ids(gt, m=m_uniform), val_dir, "img_{}.png".format(i), args.alpha)
            #print('saving ...', np.unique(pred), np.unique(gt))

        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

    val_list= np.array([[gts, preds]])
    np.save("{}/validations_tg_pred_{}".format(val_dir, args.rotated), val_list)
    score, class_iou = scores(gts, preds, n_class=args.n_classes)

    for k, v in score.items():
        print(k, v)

    for i in range(args.n_classes):
        print(i, class_iou[i])

#################
    np.save("{}/parameters_{}".format(val_dir, args.rotated), args)
    np.save("{}/result.txt".format(val_dir), np.array([score, class_iou]))    

#################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams-pix2pix-PyTorch-Evaluation')
    parser.add_argument('--model', nargs='?', type=str, default='default', 
                        help='Path to the saved model or check in code more info.')
    parser.add_argument('--dataset', nargs='?', type=str, default='cityscapes', 
                        help='Dataset to use [spherical, kitti, cityscapes, etc]')
    parser.add_argument('--img_rows', nargs='?', type=int, default=512, 
                        help='Height of the input image 512')
    parser.add_argument('--img_cols', nargs='?', type=int, default=512, 
                        help='Height of the input image 1024')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='val', 
                        help='Split of dataset to test on')
    parser.add_argument('--n_classes', nargs='?', type=int, default=3, 
                        help='number of classes')
    parser.add_argument('--mode', nargs='?', type=str, default='AtoB', 
                        help='Training from a to b folders or vice versa [\'AtoB\', \'BtoA\']')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--save', action='store_true', help='use save')
    parser.add_argument('--xy_size', type=int, default=2, help='location map size')
    parser.add_argument('--alpha', type=int, default=0, help='alpha=0, side by side elseif 0<alpha<1 overlapped ')
    parser.add_argument('--rotated', type=int, default=0, help='90, 180, 270 test on rotation')
    parser.add_argument('--norm', type=str, default='', help='linear, cycle, sigmoid to format location map(coordconv) add -no for not uniform color space eg. linear otherwise linear-no')
    #parser.add_argument('--m', type=str, default='fair', help='0 without fair space')
    parser.add_argument('--it', type=int, default=250, help='model at iteration it')
    args = parser.parse_args()
    validate(args)
           


