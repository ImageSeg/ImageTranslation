import os
import time
import torch
import numpy as np
from util import is_image_file, classify, load_img, save_img_np, decode_ids, encode_img_ids

def classids():
	img_dir = "dataset/spherical/"
	cl_dir = "dataset/spherical2/"
	#np_img =np.zeros((512,3,1024))
	image_filenames = [x for x in os.listdir(img_dir) if is_image_file(x)]
	for image_name in image_filenames:
		np_img=load_img(img_dir+image_name, (1024,512), rgb=True)
		image = encode_img_ids(np_img)
		#image = classify(np_img, img=False)
		save_img_np(decode_ids(image), cl_dir+image_name, (1024,512))
if __name__ == '__main__':
	start_time = time.time()
	classids()
	#end_time = time.time()
	print("Finished in --- %s seconds ---" % (time.time() - start_time))
