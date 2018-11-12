from os import listdir
from os.path import join
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from util import is_image_file, load_img, make_grid


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, mode, xy_size, img_size, norm, rotated=0, val=False):
        super(DatasetFromFolder, self).__init__()
        if 'no'==norm.partition('-')[2]:
            self.b ="b"
        else:
            self.b ="b_uniform"
        if mode=="AtoB":
            self.photo_path = join(image_dir, "a")
            self.sketch_path = join(image_dir, self.b)
        else:
            self.photo_path = join(image_dir, self.b)
            self.sketch_path = join(image_dir, "a")
        self.image_filenames = [x for x in listdir(self.photo_path) if is_image_file(x)]
        self.img_size = img_size
        self.val = val
        self.rgb = False if xy_size > 0 else True	#Training
        self.rotated=rotated
        self.norm = norm.partition('-')[0] if norm!='' else 'linear'

        #self.location_map = make_grid(1, img.size[0], img.size[1], norm=norm) as parameter in place of norm, so one time calculated& oft used
   
        # mean and std for Cityscapes
        CITYSCAPES_MEAN = [0.28689554, 0.32513303, 0.28389177]
        CITYSCAPES_STD = [0.18696375, 0.19017339, 0.18720214]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.photo_path, self.image_filenames[index]), self.img_size, norm=self.norm, rgb=self.rgb, rotated=self.rotated)
        input = self.transform(input)
        
        if "cityscapes" in self.photo_path:
            #target = load_img(join(self.sketch_path, self.image_filenames[index][:-15] + 'gtFine_labelIds.png'), self.img_size, rgb=True, rotated=self.rotated, norm=self.norm)
            target = load_img(join(self.sketch_path, self.image_filenames[index][:-15] + 'gtFine_color.png'), self.img_size, rgb=True, rotated=self.rotated, norm=self.norm)
        else:
            target = load_img(join(self.sketch_path, self.image_filenames[index]), self.img_size, rgb=True, rotated=self.rotated, norm=self.norm)	#(same name: image and label)
        #target = load_img(join(self.sketch_path, self.image_filenames[index][:-15] + 'gtFine_color.png'), self.img_size)
        #target = load_img(join(self.sketch_path, self.image_filenames[index][:-15] + 'gtFine_color.png'))
        if not self.val:
            target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
