# ImageTranslation
Image Segmentation and Segmented Image to Real Image (Pixel to Pixel Image Translation): comparing, analysing and using different methods based on NN e.g. pix2pix working mainly with three classes namely sky, ground and objects.

### Use model
python test.py --dataset dataset --model model --cuda
###### e.g.:
CUDA_VISIBLE_DEVICES=1 python3.6 test.py --dataset facades --model checkpoint/facades/netG_model_epoch_200.pth --cuda

### Input Image, Output Image and Labeled Image respectiv to the following images as table
Example1
<p align="center">
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/input/cmp_b0202.jpg" width="280"/>
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/facades/cmp_b0202.jpg" width="280"/>
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/gold_standard/cmp_b0202.jpg" width="280"/>
</p>
Example2
<p align="center">
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/input/cmp_b0203.jpg" width="280"/>
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/facades/cmp_b0203.jpg" width="280"/>
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/gold_standard/cmp_b0203.jpg" width="280"/>
</p>
Example3
<p align="center">
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/input/cmp_b0204.jpg" width="280"/>
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/facades/cmp_b0204.jpg" width="280"/>
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/gold_standard/cmp_b0204.jpg" width="280"/>
</p>


Source: <br>
https://github.com/mrzhu-cool/pix2pix-pytorch <br>
https://pdfs.semanticscholar.org/d533/c4d9014779178349010f325e8f0f82540da8.pdf <br>
http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_2013.pdf
