import numpy as np
import os
import torch
import torch.nn as nn
#import torchvision.transforms.functional as TF

from PIL import Image
from PIL import ImageFile
import scipy.misc as misc
from collections import Counter
import numpy.ma as ma
import math
from scipy.spatial import distance as d
from scipy.spatial import KDTree as tree

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath, img_size, norm, rgb, rotated=0):
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(filepath).convert('RGB')
    #img = img.resize(img_size)
    #if rotated!=0:
    #    img = img.rotate(rotated)
    if rgb:
        img = img.resize(img_size)
        if rotated!=0:
            img = img.rotate(rotated)
        return np.array(img)
    img = img.resize(img_size, Image.BICUBIC)
    if rotated!=0:
        img = img.rotate(rotated)
    img_dim = np.zeros((img.size[1], img.size[0], 5), dtype=np.float32)
    location_x = make_grid(1, img.size[0], img.size[1], norm=norm)[0,0,:,::1]
    location_y = make_grid(1, img.size[0], img.size[1], norm=norm)[0,1,:,::1]
    imgarr = np.array(img)
    #print('5 channels image [util/load_img(...)]')
    img_dim[:, :, 0] = imgarr[:, :, 0]
    img_dim[:, :, 1] = imgarr[:, :, 1]
    img_dim[:, :, 2] = imgarr[:, :, 2]
    img_dim[:, :, 3] = location_x.numpy().astype(np.float128).T
    img_dim[:, :, 4] = location_y.numpy().astype(np.float128).T
     
    return img_dim

def full_img(image_tensor, img_size):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    return  image_numpy

def save_img(image_tensor, filename, img_size):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    img_dim_np = np.zeros((img_size[0], img_size[1], 3))
    img_dim_np[:, :, 0] = image_numpy[:, :, 0].T
    img_dim_np[:, :, 1] = image_numpy[:, :, 1].T
    img_dim_np[:, :, 2] = image_numpy[:, :, 2].T

    img_dim_np = img_dim_np.astype(np.uint8)

    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print ("Image saved as {}".format(filename))
    
def save_img_np(image_numpy, filename, img_size):

    misc.imsave(filename, image_numpy)
    print ("Image saved as {}".format(filename))

def decode_ids(temp, m='', n_classes=3):
	colors = [[0, 255, 0],[255, 0, 0],[0, 0, 255]]
	#temp = encode_img_ids(temp)
	#colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0]]
	#colors = [[0, 255, 0],[255, 0, 0],[0, 0, 255]]#for classids()
	if m=='0':
	    colors = [[0, 0, 0],[128, 64, 128],[70, 130, 180]]
	label_colours = dict(zip(range(3), colors))
	r = temp.copy()
	g = temp.copy()
	b = temp.copy()
	for l in range(0, n_classes):
		r[temp == l] = label_colours[l][0]
		g[temp == l] = label_colours[l][1]
		b[temp == l] = label_colours[l][2]

	rgb = np.zeros((temp.shape[0], temp.shape[1], 3)).astype(np.uint8)
	rgb[:, :, 0] = r
	rgb[:, :, 1] = g
	rgb[:, :, 2] = b
	return rgb


def encode_ĺbl_ids(image_tensor, m='', n_classes=3):

	image = image_tensor.astype(np.uint8)
	classe = [[0, 0, 0], [1, 1, 1], [2,2,2]]
	colors = np.array([[0, 255, 0],[255, 0, 0],[0, 0, 255]])
	if m=='0':
		colors = np.array([[0, 0, 0],[128, 64, 128],[70, 130, 180]])
	#print(colors, np.unique(image))
	image[np.where(((image==colors[1]).all(axis=2)))] = classe[1]
	image[np.where(((image==colors[2]).all(axis=2)))] = classe[2]
	image[np.where(((image==colors[0]).all(axis=2)))] = classe[0]
	return image[:,:,0]
	
def encode_img_ids(image_tensor, n_classes=3):
	boundaries = [([245, 0, 0], [256, 10, 10]), ([0,0,245],[10, 10, 256])]
	colors = [[100, 255, 100],[255, 100, 100],[100, 100, 255]]
	classe = [[0, 0, 0], [1, 1, 1], [2,2,2]]

	label_colours = dict(zip(range(3), colors))
	if isinstance(image_tensor, torch.Tensor):
	    image_numpy = image_tensor.float().numpy()
	    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
	else:
	    image_numpy = image_tensor
	image = image_numpy.astype(np.uint8)
	out = np.zeros(image.shape).astype(np.uint8)
	out[np.where(((image<=colors[1]).all(axis=2)))] = classe[1]#2
	out[np.where(((image<=colors[2]).all(axis=2)))] = classe[2]#1
	out[np.where(((image<=colors[0]).all(axis=2)))] = classe[0]#0
	#image[np.where((((image==classe[0])&(image==classe[2])).all(axis=2)))] = classe[1]
	#print(np.unique(out[:,:,0]))
	return out[:,:,0]
	
def encode_img_ids_2(image_tensor, n_classes=3):
	boundaries = [([245, 0, 0], [256, 10, 10]), ([0,0,245],[10, 10, 256])]
	colors = [[100, 255, 100],[255, 100, 100],[100, 100, 255]]
	classe = [[0, 0, 0], [1, 1, 1], [2,2,2]]

	label_colours = dict(zip(range(3), colors))
	if isinstance(image_tensor, torch.Tensor):
	    image_numpy = image_tensor.float().numpy()
	    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
	else:
	    image_numpy = image_tensor
	image = image_numpy.astype(np.uint8)
	out = np.zeros(image.shape).astype(np.uint8)
	out[np.where(((image<=colors[1]).all(axis=2)))] = classe[2]#1
	out[np.where(((image<=colors[2]).all(axis=2)))] = classe[1]#2
	out[np.where(((image<=colors[0]).all(axis=2)))] = classe[0]#0
	#image[np.where((((image==classe[0])&(image==classe[2])).all(axis=2)))] = classe[1]
	#print(np.unique(out[:,:,0]))
	return out[:,:,0]

	
def trans_classe(image_tensor, n_classes=3):

	image_numpy = image_tensor.float().numpy()
	image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
	
	image = image_numpy.astype(np.uint8)
	
	rows = image.shape[0]
	cols = image.shape[1]
	
	out = np.empty(shape=(rows, cols, 3)).astype(np.uint8)
	for i in range(rows):
		for j in range(cols):
			out[i][j] = process_color(image[i][j])

	return out[:,:,0]
	
def process_color(pixel):
	classe = [[0, 0, 0], [1, 1, 1], [2,2,2]]
	colors = [[0, 255, 0],[255, 0, 0],[0, 0, 255]]
	#classe = np.array([[[0, 255, 0]],[[255, 0, 0]],[[0, 0, 255]]])
	
	c0= d.cdist(np.expand_dims(pixel, axis=0), classe[0], metric='euclidean') #canberra #'minkowski', p=2.
	c1= d.cdist(np.expand_dims(pixel, axis=0), classe[1], metric='euclidean')
	c2= d.cdist(np.expand_dims(pixel, axis=0), classe[2], metric='euclidean')
	
	mn=np.minimum(c0,c1,c2)	
	if c2==mn:
		px=classe[2]
	if c0==mn:
		px=classe[0]
	if c1==mn:
		px=classe[1]
	return px
	
def dist(x,y):   
    return np.array(np.sqrt(np.sum((x-y)**2)))

#input: label(3d), output: ids(2d), img for transtlated image or label
def classify(image_tensor, img=True, m=''):

	image_numpy = image_tensor.float().numpy()
	if img:
	    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
		
	rows = image_numpy.shape[0]
	cols = image_numpy.shape[1]
	#print(image_numpy.shape)
	image = image_numpy.astype(np.uint8).reshape(rows*cols, 3)
	
	
	colors = np.array([[0, 255, 0],[255, 0, 0],[0, 0, 255]])
	if m=='0':
		 colors = [[0, 0, 0],[128, 64, 128],[70, 130, 180]]
	#print(np.unique(image), colors)
	dr, classe_ids=tree(colors).query(image, p=1)

	out=classe_ids.reshape(rows, cols)

	return out

def make_grid(batch, height, width, norm):	#default = 'linear'
    height = height
    width = width
    grids_h = torch.arange(0,height).view(height,1)
    grids_h = grids_h.repeat(1, width)
    grids_h = grids_h.view(-1)
    grids_w = torch.arange(0, width).view(1,width)
    grids_w = grids_w.repeat(height,1)
    grids_w = grids_w.view(-1)
    grid = torch.stack([grids_w, grids_h], 1)
    grid = grid.view(height, width, 2)
    grid = grid.transpose(2,1).transpose(1,0)
    if norm=='linear':
        grid[0] = grid[0]*255/float(width-1)
        grid[1] = grid[1]*255/float(height-1)
    elif norm=='sigmoid':
        sgm = nn.Sigmoid()
        grid[0] = grid[0]*24/float(width-1)-12.0
        grid[1] = grid[1]*24./float(height-1)-12.0
        grid = 255*sgm(grid)
    elif norm=='cycle':
        grid_np = make_cycle_grid(width, height)
        grid = torch.from_numpy(grid_np)
        #print(grid.shape)
    elif norm=='c_linear':
        grid_np = cycle_xy(width, height, norm)
        grid = torch.from_numpy(grid_np)
    elif norm=='c_sin':
        grid_np = cycle_xy(width, height, norm)
        grid = torch.from_numpy(grid_np)   
    grid = grid.unsqueeze(0)
    grid = grid.expand(batch, 2, height, width)
    return grid
    
def make_cycle_grid(width, height):
	if width!=height or width%2==1 or height%2==1:
		raise RuntimeError('ERROR: width is not equal height or one of them is not divisible 2 for cycle_location_map')
	h = int(math.floor(height/2))	#not working
	w = int(math.trunc(width/2))	#not working
	
	col = np.flip(range(-w,h,1))
	row = range(-w,h,1)
	map_ = [np.abs(c)+np.abs(row) for c in col]
	x_map = np.array(map_).reshape(width,height)/width
	y_map = (-1)*x_map + 1
	return np.array([x_map, y_map])
	  
def cycle_xy(width, height, norm, norm_=255):	#mode=='c_linear'
    #Width
    w_i= int(width/2)
    w_e= int((width+1)/2)
    #Height
    h_i = int(height/2)
    h_e = int((height+1)/2)

    if w_i==w_e:    #width
        b = range(-w_i, -1)
        b_ = range(2, w_e+1)
        _o = np.hstack((b,[-0.342, 0.30], b_))
    else:
        _o = np.array(range(-w_i,w_e,1))
    if norm=='c_sin':
        _o=np.sin(0.5*np.pi*_o/w_i)
        _ow = np.abs(_o)
    else:
        _ow = np.abs(_o) /w_i

    if width==height:   #simply for tuning
        _oh = _ow.T
    else:   #height
        if h_i==h_e:
            b = range(-h_i, -1)
            b_ = range(2, h_e+1)
            _o = np.hstack((b,[-0.36, 0.26], b_))
        else:
            _o = np.array(range(-h_i,h_e,1))
        if norm=='c_sin':
            _o=np.sin(0.5*np.pi*_o/h_i)
            _oh = np.abs(_o)
        else:
            _oh = np.abs(_o)/h_i
    x_map = np.mean(np.meshgrid(norm_*_ow, norm_*_oh), axis=0)
    y_map = (-1)*x_map + norm_
    return np.array([x_map, y_map])
 
def save_val(seg, gt, f, n, alpha):
    seg = Image.fromarray(seg)
    gt = Image.fromarray(gt)

    if alpha>0 and alpha<1:
        image_to_save = Image.blend(seg, gt, alpha)
    else:
        (width1, height1) = seg.size
        (width2, height2) = gt.size

        result_width = width1 + width2
        result_height = max(height1, height2)
        image_to_save = Image.new('RGB', (result_width, result_height))
        image_to_save.paste(im=seg, box=(0, 0))
        image_to_save.paste(im=gt, box=(width1, 0))
		
    if not os.path.exists(f):
        os.mkdir(f)
    image_to_save.save("{}/{}".format(f, n)) 
    
#startThanks to:
#https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
#https://pypi.python.org/pypi/fcn/6.1.4
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    #print(np.bincount(n_class * label_true[mask].astype(int) +label_pred[mask], minlength=n_class ** 2))
    hist = np.bincount(n_class * label_true[mask].astype(int) +label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
	"""Returns accuracy score evaluation result.
	  - overall accuracy
	  - mean accuracy
	  - mean IU
	  - fwavacc
	"""
	hist = np.zeros((n_class, n_class))
	for lt, lp in zip(label_trues, label_preds):
		hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
	acc = np.diag(hist).sum() / hist.sum()
	acc_cls = np.diag(hist) / hist.sum(axis=1)
	acc_cls = np.nanmean(acc_cls)
	iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
	mean_iu = np.nanmean(iu)
	freq = hist.sum(axis=1) / hist.sum()
	fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
	cls_iu = dict(zip(range(n_class), iu))
	#return acc, acc_cls, mean_iu, fwavacc
	return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,}, cls_iu
#endThanks

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    else:
        print(net.parameters())
    print(name)
    print('mean: ', mean, 'and ', count, ' parameters')

def rm_2channel(img5d, sz):
    #sz = img5d.size
    img3d = np.zeros((sz[1], sz[0], 3), dtype=np.float32)
    img3d[:, :, 0] = img5d[:, :, 0]
    img3d[:, :, 1] = img5d[:, :, 1]
    img3d[:, :, 2] = img5d[:, :, 2]
    return img3d

if __name__=="__main__":
	grid = make_grid(1, 6, 6, norm='cycle')
	print(grid, grid.shape)
	#np.save('location_map_cycle', grid.numpy())
	#loc = make_cycle_grid(width=5, height=5)
	#print(loc)
	#for c in col:
	#	cycle_map = np.concatenate((cycle_map, np.abs(c)+np.abs(row)), axis=0)
	#print(cycle_map.reshape(w,h))
