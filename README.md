# Image Segmentation using Pix2pix with Coordconv Normalized Color Space <!-- Image Translation  with Conditional Adversarial Networks -->

Image Segmentation as Pixel to Pixel Image Translation: comparing, analysing and using different methods based on NN e.g. pix2pix working mainly with three classes namely sky, ground and objects.

### Use model
python test.py --dataset dataset --model cityscapes --cuda --morm cycle  --m fair
<!-- ###### e.g.:
CUDA_VISIBLE_DEVICES=1 python3.6 test.py --dataset facades --model checkpoint/facades/netG_model_epoch_200.pth --cuda -->

### Final output Image, Input Image and coorconv and normalized output Image respectiv to the following images as table
Example1
<p align="center">
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/cityscapes/berlin_000000_000019_leftImg8bit.png" width="280"/>
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/cityscapes/o_berlin_000000_000019_leftImg8bit.png" width="280"/>
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/cityscapes/t_berlin_000000_000019_leftImg8bit.png" width="280"/>
</p>
Example2
<p align="center">
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/cityscapes/berlin_000001_000019_leftImg8bit.png" width="280"/>
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/cityscapes/o_berlin_000001_000019_leftImg8bit.png" width="280"/>
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/cityscapes/t_berlin_000001_000019_leftImg8bit.png" width="280"/>
</p>
Example3
<p align="center">
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/cityscapes/munich_000028_000019_leftImg8bit.png" width="280"/>
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/cityscapes/o_munich_000028_000019_leftImg8bit.png" width="280"/>
  <img src="https://github.com/ImageSeg/ImageTranslation/blob/master/result/cityscapes/t_munich_000028_000019_leftImg8bit.png" width="280"/>
</p>


Source: <br>
https://github.com/mrzhu-cool/pix2pix-pytorch <br>
https://pdfs.semanticscholar.org/d533/c4d9014779178349010f325e8f0f82540da8.pdf <br>
http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_2013.pdf
